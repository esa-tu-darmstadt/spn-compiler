#include "conversion.hpp"

#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/TypeSwitch.h"


namespace spn::lo2hw::conversion {

void ConversionHelper::createHwOps() {
  std::vector<PortInfo> binPorts{
    inPort("clk", sigType),
    inPort("rst", sigType),
    inPort("in_a", probType),
    inPort("in_b", probType),
    outPort("out_c", probType)
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

  hwOps["sv_add"] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_add"),
    ArrayRef<PortInfo>(binPorts)
  );

  hwOps["sv_mul"] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_mul"),
    ArrayRef<PortInfo>(binPorts)
  );

  hwOps["sv_categorical"] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_categorical"),
    ArrayRef<PortInfo>(catPorts)
  );

  hwOps["sv_constant"] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_constant"),
    ArrayRef<PortInfo>(constPorts)
  );

  hwOps["sv_log"] = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_log"),
    ArrayRef<PortInfo>(logPorts)
  );
}

int64_t ConversionHelper::getDelay(const std::string& name) const {
  // just some arbitrary values
  if (name == "sv_add") {
    return 5;
  } else if (name == "sv_mul") {
    return 10;
  } else if (name == "sv_categorical") {
    return 1;
  } else if (name == "sv_constant") {
    return 0;
  } else if (name == "sv_log") {
    return 10;
  }

  throw std::runtime_error("Unknown module type!");
}

void ConversionHelper::assignInstanceIds(ModuleOp root) {
  uint64_t id = 0;

  root.walk([&](Operation *op) {
    instanceIds[op] = id++;
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

      if (!newOperand) {
        throw std::runtime_error("Mapping to new operand does not exist!");
      }

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
    Operation *hwExtMod = llvm::TypeSwitch<Operation *, Operation *>(op)
      .Case<SPNAdd>([&](SPNAdd op) { return helper.getMod("sv_add"); })
      .Case<SPNMul>([&](SPNMul op) { return helper.getMod("sv_mul"); })
      .Case<SPNCategoricalLeaf>([&](SPNCategoricalLeaf op) { return helper.getMod("sv_categorical"); })
      .Case<SPNConstant>([&](SPNConstant op) { return helper.getMod("sv_constant"); })
      .Case<SPNLog>([&](SPNLog op) { return helper.getMod("sv_log"); })
      .Default([&](Operation *op) -> InstanceOp {
        throw std::runtime_error("Unexpected type");
      });

    InstanceOp newInstance = builder.create<InstanceOp>(
      loc,
      hwExtMod,
      instanceName,
      operandsRef
    );

    // the old result value now points to the new result value
    mapping.map(op->getResults(), newInstance.getResults());
  });

  return modOp;
}

ModuleOp convert(ModuleOp root) {
  ConversionHelper helper(root.getContext());
  helper.assignInstanceIds(root);
  std::vector<HWModuleOp> modOps;

  root.walk([&](SPNBody body) {
    Optional<HWModuleOp> modOp = createBodyModule(body, helper);

    if (!modOp.has_value())
      throw std::runtime_error("Could not create body module!");

    modOps.push_back(modOp.value());
  });

  // put everything together
  OpBuilder builder(root.getContext());

  ModuleOp newRoot = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&newRoot.getRegion().front());
  builder.insert(helper.getMod("sv_add"));
  builder.insert(helper.getMod("sv_mul"));
  builder.insert(helper.getMod("sv_log"));
  builder.insert(helper.getMod("sv_constant"));
  builder.insert(helper.getMod("sv_categorical"));
  
  for (auto op : modOps)
    builder.insert(op);

  assert(succeeded(newRoot.verify()));

  // schedule problem and insert buffers
  SchedulingProblem problem(newRoot.getOperation());
  schedule(newRoot, helper, problem);

  // from here on the code gets ugly
  insertShiftRegisters(newRoot, helper, problem);

  //dumpAsDOT(problem, llvm::outs());

  return newRoot;
}

void schedule(ModuleOp root, ConversionHelper& helper, SchedulingProblem& problem) {
  // inspired by Scheduling.cpp

  using namespace ::circt::scheduling;
  using namespace ::llvm;
  using OperatorType = SchedulingProblem::OperatorType;

  // construct problem
  for (const std::string& name : std::vector<std::string>{"sv_add", "sv_mul", "sv_constant", "sv_categorical", "sv_log"}) {
    OperatorType opType = problem.getOrInsertOperatorType(name);
    int64_t delay = helper.getDelay(name);
    problem.setLatency(opType, delay);
  }

  root.walk([&](Operation *op) {
    // check that this operation is relevant
    InstanceOp instOp = dyn_cast<InstanceOp>(op);

    if (!instOp)
      return;

    Operation *refMod = instOp.getReferencedModule(nullptr);

    HWModuleExternOp extOp = dyn_cast<HWModuleExternOp>(refMod);
    std::string instName = instOp.getName().getValue().str();
    std::string modName = extOp.getName().str();

    problem.insertOperation(op);  
    
    int64_t delay = helper.getDelay(modName);

    // add dependencies
    for (Value operand : op->getOperands()) {
      if (!operand.getDefiningOp())
        continue;
      
      problem.insertDependence(std::make_pair(operand.getDefiningOp(), op));
    }

    // say what operator type we have
    problem.setLinkedOperatorType(op, problem.getOrInsertOperatorType(modName));
  });

  if (failed(problem.check()))
    throw std::runtime_error("Problem check failed!");

  if (failed(scheduleASAP(problem)))
    throw std::runtime_error("Could not schedule problem!");
}

void insertShiftRegisters(ModuleOp root, ConversionHelper& helper, SchedulingProblem& problem) {
  OpBuilder builder(root.getContext());

  std::vector<std::tuple<
    Operation *,  // from
    Operation *,  // to
    uint32_t      // delay
  >> jobs;

  // first collect all the delays we want to insert
  root.walk([&](InstanceOp op) {
    for (Value operand : op.getOperands()) {
      Operation *defOp = operand.getDefiningOp();

      // is a block argument or something
      if (!defOp)
        continue;

      assert(problem.getStartTime(op).has_value());
      assert(problem.getLatency(problem.getLinkedOperatorType(op).value()).has_value());
      assert(problem.getStartTime(defOp).has_value());

      uint32_t meStartTime = problem.getStartTime(op).value();
      uint32_t defOpLatency = problem.getLatency(problem.getLinkedOperatorType(defOp).value()).value();
      uint32_t defOpStartTime = problem.getStartTime(defOp).value();

      assert(defOpStartTime + defOpLatency <= meStartTime);

      uint32_t delay = meStartTime - (defOpStartTime + defOpLatency);

      if (delay == 0)
        continue;

      jobs.emplace_back(
        defOp,
        op.getOperation(),
        delay
      );
    }
  });

  Value clk, rst;

  // stupid way to find the clk and rst signals
  root.walk([&](HWModuleOp op) {
    Block *body = op.getBodyBlock();
    clk = body->getArguments()[0];
    rst = body->getArguments()[1];
    return WalkResult::interrupt();
  });

  auto delaySignal = [&](Value input, uint32_t delay) -> std::tuple<Value, Operation *> {
    assert(delay >= 1);
    
    builder.setInsertionPointAfter(input.getDefiningOp());
    Value prev = input;
    Operation *ignoreOp = nullptr;

    for (uint32_t i = 0; i < delay; ++i) {
      CompRegOp reg = builder.create<CompRegOp>(
        builder.getUnknownLoc(), std::move(prev), clk, "someReg"
      );
      prev = reg.getResult();

      if (!ignoreOp)
        ignoreOp = reg;
    }

    return std::make_tuple(prev, ignoreOp);
  };

  // finally insert delays
  for (auto [from, to, delay] : jobs) {
    Value result = from->getResults()[0];
    auto [delayedResult, ignoreOp] = delaySignal(result, delay);
    // we introduce a new usage which we need to ignore
    result.replaceAllUsesExcept(delayedResult, ignoreOp);
  }
}

// TODO: See what lowering to systemverilog does to the generated hw module!
void test(MLIRContext *ctxt) {
  OpBuilder builder(ctxt);
  ConversionHelper helper(ctxt);

  ModuleOp root = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&root.getRegion().front());

  std::vector<PortInfo> ports{
    helper.inPort("clk", builder.getI1Type()),
    helper.inPort("rst", builder.getI1Type()),
    helper.outPort("result", builder.getI32Type())
  };

  HWModuleOp mod = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("hw_mod"),
    ArrayRef<PortInfo>(ports)
  );

  // remove old hw output
  mod.getBodyBlock()->clear();

  Value clk = mod.getBodyBlock()->getArguments()[0];

  builder.setInsertionPointToStart(mod.getBodyBlock());

  ConstantOp constOp = builder.create<ConstantOp>(
    builder.getUnknownLoc(),
    builder.getI32Type(),
    123456
  );

  ConstantOp secondClk = builder.create<ConstantOp>(
    builder.getUnknownLoc(),
    builder.getI1Type(),
    0
  );

  CompRegOp reg = builder.create<CompRegOp>(
    builder.getUnknownLoc(),
    constOp.getResult(),
    clk,
    //builder.getStringAttr("someReg")
    "someReg"
  );

  builder.setInsertionPointToStart(mod.getOperation()->getBlock());
  
  std::vector<StringRef> requiredAttributes{};
  
  HWGeneratorSchemaOp schema = builder.create<HWGeneratorSchemaOp>(
    builder.getUnknownLoc(),
    "SOME_SCHEMA",
    "lalalala",
    builder.getStrArrayAttr(ArrayRef<StringRef>(requiredAttributes))
  );

  /*
  std::vector<PortInfo> ports2;

  builder.create<HWModuleGeneratedOp>(
    builder.getUnknownLoc(),
    SymbolRefAttr::get(schema),
    builder.getStringAttr("SomeGeneratedModule"),
    ArrayRef<PortInfo>(ports2)
  );
   */

  builder.create<OutputOp>(
    builder.getUnknownLoc(),
    reg.getResult()
  );

  root.dump();
}

}