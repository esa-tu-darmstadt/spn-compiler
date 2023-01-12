#include "LoSPNtoFPGA/conversion.hpp"

#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/TypeSwitch.h"


namespace mlir::spn::fpga {

static const char *SV_ADD = "FPAdd";
static const char *SV_MUL = "FPMult";
static const char *SV_LOG = "FPLog";
static const char *SV_CONSTANT = "sv_constant";
static const char *SV_CATEGORICAL = "sv_categorical";

void ConversionHelper::createHwOps() {
  std::vector<PortInfo> binPorts{
    inPort("clock", sigType),
    inPort("reset", sigType),
    inPort("io_a", probType),
    inPort("io_b", probType),
    outPort("io_r", probType)
  };

  std::vector<PortInfo> catPorts{
    inPort("clk", sigType),
    inPort("rst", sigType),
    inPort("in_index", indexType),
    outPort("out_prob", probType)
  };

  std::vector<PortInfo> constPorts{
    inPort("clk", sigType),
    inPort("rst", sigType),
    outPort("out_const", probType)
  };

  std::vector<PortInfo> logPorts{
    inPort("clk", sigType),
    inPort("rst", sigType),
    inPort("in_a", probType),
    outPort("out_b", probType)
  };

  std::vector<PortInfo> bodyPorts{
    inPort("clk", sigType),
    inPort("rst", sigType),
    outPort("out_prob", probType)
  };

  hwOps[SV_ADD] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr(SV_ADD),
    ArrayRef<PortInfo>(binPorts)
  );

  hwOps[SV_MUL] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr(SV_MUL),
    ArrayRef<PortInfo>(binPorts)
  );

  hwOps[SV_CATEGORICAL] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr(SV_CATEGORICAL),
    ArrayRef<PortInfo>(catPorts)
  );

  hwOps[SV_CONSTANT] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr(SV_CONSTANT),
    ArrayRef<PortInfo>(constPorts)
  );

  hwOps[SV_LOG] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr(SV_LOG),
    ArrayRef<PortInfo>(logPorts),
    SV_LOG
  );

  //hwOps[SV_LOG]->setAttr("filenames", builder.getStringAttr("somefile.sv"));
}

int64_t ConversionHelper::getDelay(const std::string& name) const {
  // just some arbitrary values
  if (name == SV_ADD) {
    return 5;
  } else if (name == SV_MUL) {
    return 10;
  } else if (name == SV_CATEGORICAL) {
    return 1;
  } else if (name == SV_CONSTANT) {
    return 0;
  } else if (name == SV_LOG) {
    return 10;
  }

  assert(false && "Unknown module type!");
}

void ConversionHelper::assignInstanceIds(ModuleOp root) {
  uint64_t id = 0;

  root.walk([&](Operation *op) {
    instanceIds[op] = id++;
  });
}

void ConversionHelper::assignCatModules(ModuleOp root) {
  root.walk([&](SPNCategoricalLeaf leaf) {
    HWModuleOp catOp = createCategoricalModule(*this, leaf, getInstanceId(leaf)).value();
    catModules[leaf.getOperation()] = catOp.getOperation();
  });
}

// TODO: Decide if we want to utilize Optional or throw exceptions!
Optional<HWModuleOp> createBodyModule(SPNBody body, ConversionHelper& helper) {
  OpBuilder builder(helper.getContext());

  // build top hw module
  std::vector<PortInfo> modOpPorts{
    helper.inPort("clk", helper.getSigType()),
    helper.inPort("rst", helper.getSigType()),
    helper.port("out_prob", PortDirection::OUTPUT, helper.getProbType())
  };

  for (std::size_t i = 0; i < body.getOperands().size(); ++i)
    modOpPorts.push_back(helper.port(
      "in_" + std::to_string(i), PortDirection::INPUT, helper.getIndexType()
    ));

  HWModuleOp modOp = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("spn_body"),
    ArrayRef<PortInfo>(modOpPorts)
  );

  // remove the default hw.output
  modOp.getBodyBlock()->front().erase();
  Value modClk = modOp.getRegion().front().getArguments()[0];
  Value modRst = modOp.getRegion().front().getArguments()[1];

  builder.setInsertionPointToStart(modOp.getBodyBlock());

  // use this to map the old values to the new values
  BlockAndValueMapping mapping;

  // remap the block arguments
  mapping.map(
    body.getRegion().front().getArguments(),
    // drop clk and rst signals
    modOp.getRegion().front().getArguments().drop_front(2)
  );

  body.walk([&](Operation *op) {
    // if op is the body itself ignore it
    if (llvm::isa<SPNBody>(op))
      return;

    // get the new operands
    std::vector<Value> operands{
      modClk, modRst
    };

    for (Value operand : op->getOperands()) {
      Value newOperand = mapping.lookupOrNull(operand);
      assert(newOperand);

      operands.push_back(newOperand);
    }
    ArrayRef<Value> operandsRef(operands);

    Location loc = op->getLoc();
    StringAttr instanceName = builder.getStringAttr(helper.getInstanceName(op));

    // SPNYield forms an exception
    if (SPNYield yield = llvm::dyn_cast<SPNYield>(op)) {
      builder.create<OutputOp>(
        loc,
        ValueRange(operandsRef.drop_front(2))
      );

      return;
    }

    // get the external module which we refer to in the new instance
    Operation *mod = llvm::TypeSwitch<Operation *, Operation *>(op)
      .Case<SPNAdd>([&](SPNAdd op) { return helper.getMod(SV_ADD); })
      .Case<SPNMul>([&](SPNMul op) { return helper.getMod(SV_MUL); })
      .Case<SPNCategoricalLeaf>([&](SPNCategoricalLeaf op) {
        return helper.getCatModule(op);
      })
      .Case<SPNConstant>([&](SPNConstant op) {
        float f32 = op.getValue().convertToDouble();
        uint32_t bits = *reinterpret_cast<const uint32_t *>(&f32);

        // TODO: Conversion must be configurable!
        return builder.create<ConstantOp>(
          builder.getUnknownLoc(),
          helper.getProbType(),
          bits
        );
      })
      .Case<SPNLog>([&](SPNLog op) { return helper.getMod(SV_LOG); })
      .Default([&](Operation *op) -> InstanceOp {
        assert(false && "Unexpected type");
      });

    Operation *newInstance = nullptr;
    
    if (llvm::isa<ConstantOp>(mod)) {
      newInstance = mod;
    } else {
      newInstance = builder.create<InstanceOp>(
        loc,
        mod,
        instanceName,
        operandsRef
      ).getOperation();
    }

    // the old result value now points to the new result value
    mapping.map(op->getResults(), newInstance->getResults());
  });

  return modOp;
}

Optional<ModuleOp> convert(ModuleOp root) {
  ConversionHelper helper(root.getContext());
  helper.assignInstanceIds(root);
  helper.assignCatModules(root);
  std::vector<HWModuleOp> modOps;

  root.walk([&](SPNBody body) {
    Optional<HWModuleOp> modOp = createBodyModule(body, helper);

    if (!modOp.has_value()) {
      modOps.clear();
      return WalkResult::interrupt();
    }

    modOps.push_back(modOp.value());
  });

  // fails if one module could not be converted or none were found
  if (modOps.size() == 0)
    return Optional<ModuleOp>();

  // put everything together
  OpBuilder builder(root.getContext());

  ModuleOp newRoot = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&newRoot.getRegion().front());

  builder.create<VerbatimOp>(
    builder.getUnknownLoc(),
    helper.getVerilogIncludeString()
  );

  builder.insert(helper.getMod(SV_ADD));
  builder.insert(helper.getMod(SV_MUL));
  builder.insert(helper.getMod(SV_LOG));
  builder.insert(helper.getMod(SV_CONSTANT));
  
  for (auto e : helper.getCatModules())
    builder.insert(std::get<1>(e));
  
  for (auto op : modOps)
    builder.insert(op);

  assert(succeeded(newRoot.verify()));

  // schedule problem and insert buffers
  for (auto modOp : modOps) {
    scheduling::SchedulingProblem problem(modOp.getOperation());

    problem.construct();

    if (failed(problem.check()))
      assert(false && "Problem check failed!");

    if (failed(scheduleASAP(problem)))
      assert(false && "Could not schedule problem!");

    // from here on the code gets ugly
    problem.insertDelays();
  }

  return newRoot;
}

Optional<HWModuleOp> createCategoricalModule(ConversionHelper& helper, SPNCategoricalLeaf op, uint64_t id) {
  
  OpBuilder builder = helper.getBuilder();

  std::vector<PortInfo> catPorts{
    helper.inPort("clk", helper.getSigType()),
    helper.inPort("rst", helper.getSigType()),
    helper.inPort("in_index", helper.getIndexType()),
    helper.outPort("out_prob", helper.getProbType())
  };

  HWModuleOp catOp = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("spn_categorical_" + std::to_string(id)),
    ArrayRef<PortInfo>(catPorts)
  );

  // remove the default hw.output
  catOp.getBodyBlock()->front().erase();
  Value modClk = catOp.getRegion().front().getArguments()[0];
  Value modRst = catOp.getRegion().front().getArguments()[1];

  builder.setInsertionPointToStart(catOp.getBodyBlock());

  // create probability array
  std::vector<Value> probabilities;

  for (Attribute prob : op.getProbabilities()) {
    float f32 = float(llvm::dyn_cast<FloatAttr>(prob).getValueAsDouble());
    uint32_t bits = *reinterpret_cast<const uint32_t *>(&f32);

    // TODO: Conversion must be configurable!
    ConstantOp constant = builder.create<ConstantOp>(
      builder.getUnknownLoc(),
      helper.getProbType(),
      bits
    );

    probabilities.push_back(constant.getResult());
  }

  ArrayCreateOp arrayOp = builder.create<ArrayCreateOp>(
    builder.getUnknownLoc(),
    ValueRange(probabilities)
  );

  // The contraints require that the input bit width is just large enough
  // to index all elements.
  Value indexOp = catOp.getRegion().front().getArguments()[2];
  unsigned bitWidth = llvm::APInt(32, probabilities.size()).ceilLogBase2();
  
  ExtractOp extractOp = builder.create<ExtractOp>(
    builder.getUnknownLoc(),
    indexOp,
    0,
    bitWidth
  );

  ArrayGetOp getOp = builder.create<ArrayGetOp>(
    builder.getUnknownLoc(),
    arrayOp,
    extractOp.getResult()
  );

  builder.create<OutputOp>(
    builder.getUnknownLoc(),
    ValueRange(std::vector<Value>{getOp.getResult()})
  );

  return catOp;
}

Optional<HWModuleOp> createHistogramModule(ConversionHelper& helper, SPNHistogramLeaf op, uint64_t id) {
  
}

}