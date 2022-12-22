#include "conversion.hpp"

#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/TypeSwitch.h"


namespace spn::lo2hw::conversion {

void ConversionHelper::createHwOps() {
  std::vector<PortInfo> binPorts{
    inPort("in_a", probType),
    inPort("in_a", probType),
    outPort("out_c", probType)
  };

  std::vector<PortInfo> catPorts{
    inPort("in_index", indexType),
    outPort("out_prob", probType)
  };

  std::vector<PortInfo> constPorts{
    outPort("out_const", probType)
  };

  std::vector<PortInfo> logPorts{
    inPort("in_a", probType),
    outPort("out_b", probType)
  };

  std::vector<PortInfo> bodyPorts{
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

  builder.setInsertionPointToStart(modOp.getBodyBlock());

  // use this to map the old values to the new values
  BlockAndValueMapping mapping;

  // remap the block arguments
  mapping.map(
    body.getRegion().front().getArguments(),
    modOp.getRegion().front().getArguments()
  );

  body.walk([&](Operation *op) {
    // if op is the body itself ignore it
    if (llvm::isa<SPNBody>(op))
      return;

    // get the new operands
    std::vector<Value> operands;
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
        ValueRange(operands)
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

  schedule(newRoot, helper);

  return newRoot;
}

void schedule(ModuleOp root, ConversionHelper& helper) {
  // inspired by Scheduling.cpp
  
  using namespace ::circt::scheduling;
  using namespace ::llvm;
  using OperatorType = SchedulingProblem::OperatorType;

  //SPNBody body = root.getRegion().front().getOperations()[5];

  SchedulingProblem problem;
  problem.setContainingOp(root.getOperation());

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

    // ignore constants
    Operation *refMod = instOp.getReferencedModule(nullptr);

    //if (refMod == helper.getMod("sv_constant")) {
    //  llvm::outs() << "ignoring constant\n";
    //  return;
    //}

    HWModuleExternOp extOp = dyn_cast<HWModuleExternOp>(refMod);
    std::string instName = instOp.getName().getValue().str();
    std::string modName = extOp.getName().str();

    llvm::outs() << "inserting operation " << instName << " (" << modName << ")" << "\n";
    problem.insertOperation(op);  
    
    int64_t delay = helper.getDelay(modName);
    llvm::outs() << "operator has delay " << delay << "\n";

    // add dependencies
    for (Value operand : op->getOperands()) {
      if (!operand.getDefiningOp())
        continue;
      
      
      llvm::outs() << "adding dependency\n";
      problem.insertDependence(std::make_pair(operand.getDefiningOp(), op));
    }

    // say what operator type we have
    problem.setLinkedOperatorType(op, problem.getOrInsertOperatorType(modName));
  });

  llvm::outs() << "Checking problem...\n";
  problem.check();

  //dumpAsDOT(problem, llvm::outs());

  if (failed(scheduleASAP(problem)))
    llvm::errs() << "Could not schedule problem!\n";
  else
    llvm::outs() << "Problem scheduled!\n";


  dumpAsDOT(problem, llvm::outs());
}

}