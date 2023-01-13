#include "LoSPNtoFPGA/conversion.hpp"

#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/TypeSwitch.h"


namespace mlir::spn::fpga {

std::vector<PortInfo> ConversionHelper::hwPorts(const std::vector<OperatorPortInfo>& ports) {
  std::vector<PortInfo> newPorts;

  for (const OperatorPortInfo& portInfo : ports) {
    PortDirection direction =
      portInfo.direction == INPUT ? PortDirection::INPUT : PortDirection::OUTPUT;

    Type type;
    switch (portInfo.type) {
      case PORT_SIGNAL: type = targetTypes.getSignalType(); break;
      case PORT_INDEX: type = targetTypes.getIndexType(); break;
      case PORT_PROBABILITY: type = targetTypes.getProbType(); break;
    }

    PortInfo newPort{
      .name = builder.getStringAttr(portInfo.name),
      .direction = direction,
      .type = type
    };

    newPorts.push_back(newPort);
  }

  return newPorts;
}

void ConversionHelper::createHwOps() {
  for (OperatorType opType : {TYPE_ADD, TYPE_MUL, TYPE_LOG}) {
    std::vector<PortInfo> ports = hwPorts(opMapping.getOperatorPorts(opType));
    std::string name = opMapping.getTypeBaseName(opType);

    hwOps[opType] = builder.create<HWModuleExternOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr(name),
      ArrayRef<PortInfo>(ports)
    );
  }
}

void ConversionHelper::assignInstanceIds(ModuleOp root) {
  uint64_t id = 0;

  root.walk([&](Operation *op) {
    instanceIds[op] = id++;
  });
}

void ConversionHelper::assignLeafModules(ModuleOp root) {
  root.walk([&](SPNCategoricalLeaf leaf) {
    HWModuleOp catOp = createLeafModule(leaf.getOperation()).value();
    leafModules[leaf.getOperation()] = catOp.getOperation();
  });
}

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

    // map op type
    OperatorType opType = OperatorTypeMapping::getOperatorType(op);

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
    if (opType == TYPE_YIELD) {
      builder.create<OutputOp>(
        loc,
        ValueRange(operandsRef.drop_front(2))
      );

      return;
    }

    // get the external module which we refer to in the new instance
    Operation *newInstance = nullptr;

    if (opType == TYPE_CONSTANT) {
      uint32_t bits = helper.targetTypes.convertProb(
        llvm::dyn_cast<SPNConstant>(op).getValue().convertToDouble()
      );

      newInstance = builder.create<ConstantOp>(
        builder.getUnknownLoc(),
        helper.targetTypes.getProbType(),
        bits
      );
    } else {
      Operation *refMod = (opType == TYPE_CATEGORICAL || opType == TYPE_HISTOGRAM) ?
        helper.getLeafModule(op) : helper.getMod(opType);

      newInstance = builder.create<InstanceOp>(
        loc,
        refMod,
        instanceName,
        operandsRef
      ).getOperation();
    }

    helper.opMapping.setType(newInstance, opType);
    // the old result value now points to the new result value
    mapping.map(op->getResults(), newInstance->getResults());
  });

  return modOp;
}

Optional<ModuleOp> convert(ModuleOp root) {
  ConversionHelper helper(root.getContext());
  helper.assignInstanceIds(root);
  helper.assignLeafModules(root);
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

  for (OperatorType opType : {TYPE_ADD, TYPE_MUL, TYPE_LOG})
    builder.insert(helper.getMod(opType));

  for (auto e : helper.getLeafModules())
    builder.insert(std::get<1>(e));
  
  for (auto op : modOps)
    builder.insert(op);

  assert(succeeded(newRoot.verify()));

  std::error_code ec;
  llvm::raw_fd_ostream scheduleFile("schedule_info.json", ec);
  assert(!ec && "Could not open file!");

  // schedule problem and insert buffers
  for (auto modOp : modOps) {
    scheduling::SchedulingProblem problem(modOp.getOperation(), helper.opMapping);

    problem.construct();

    if (failed(problem.check()))
      assert(false && "Problem check failed!");

    if (failed(scheduleASAP(problem)))
      assert(false && "Could not schedule problem!");

    // from here on the code gets ugly
    problem.insertDelays();

    problem.writeScheduling(scheduleFile);
  }

  return newRoot;
}

Optional<HWModuleOp> ConversionHelper::createLeafModule(Operation *op) {
  // we need a new builder
  OpBuilder builder(ctxt);

  std::vector<PortInfo> catPorts = hwPorts(opMapping.getOperatorPorts(TYPE_CATEGORICAL));
  uint64_t id = getInstanceId(op);

  HWModuleOp leafOp = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr(opMapping.getFullModuleName(TYPE_CATEGORICAL, id)),
    ArrayRef<PortInfo>(catPorts)
  );

  // remove the default hw.output
  leafOp.getBodyBlock()->front().erase();
  builder.setInsertionPointToStart(leafOp.getBodyBlock());

  // create probability array
  std::vector<Value> probabilities = createProbabilityArray(builder, op);

  ArrayCreateOp arrayOp = builder.create<ArrayCreateOp>(
    builder.getUnknownLoc(),
    ValueRange(probabilities)
  );

  // The contraints require that the input bit width is just large enough
  // to index all elements.
  Value indexOp = leafOp.getRegion().front().getArguments()[2];
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

  return leafOp;
}

std::vector<Value> ConversionHelper::createProbabilityArray(OpBuilder& builder, Operation *op) {
  ArrayAttr probs;

  // TODO: Histogram
  if (SPNCategoricalLeaf leaf = llvm::dyn_cast<SPNCategoricalLeaf>(op))
    probs = leaf.getProbabilities();

  std::vector<Value> probabilities;

  for (Attribute prob : probs) {
    uint32_t bits = targetTypes.convertProb(
      llvm::dyn_cast<FloatAttr>(prob).getValueAsDouble()
    );

    ConstantOp constant = builder.create<ConstantOp>(
      builder.getUnknownLoc(),
      targetTypes.getProbType(),
      bits
    );

    probabilities.push_back(constant.getResult());
  }

  return probabilities;
}

}