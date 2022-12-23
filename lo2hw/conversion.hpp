#pragma once

#include <unordered_map>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "HiSPN/HiSPNDialect.h"

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"


using namespace ::mlir;
using namespace ::circt::hw;
using namespace ::mlir::spn::low;
using namespace ::mlir::spn::high;


namespace spn::lo2hw::conversion {

class ConversionHelper {
  MLIRContext *ctxt;
  OpBuilder builder;

  Type indexType, probType, sigType;

  std::unordered_map<Operation *, uint64_t> instanceIds;
  std::unordered_map<std::string, Operation *> hwOps;

  void createHwOps();
public:


  ConversionHelper(MLIRContext *ctxt): ctxt(ctxt), builder(ctxt) {
    indexType = builder.getI8Type();
    probType = builder.getI64Type();
    sigType = builder.getI1Type();

    createHwOps();
  }

  MLIRContext *getContext() const { return ctxt; }
  Type getIndexType() const { return indexType; }
  Type getProbType() const { return probType; }
  Type getSigType() const { return sigType; }

  PortInfo port(const std::string& name, PortDirection direction, Type type) {
    return PortInfo{
      .name = builder.getStringAttr(name),
      .direction = direction,
      .type = type
    };
  };

  PortInfo inPort(const std::string& name, Type type) { return port(name, PortDirection::INPUT, type); }
  PortInfo outPort(const std::string& name, Type type) { return port(name, PortDirection::OUTPUT, type); }

  Operation *getMod(const std::string& name) const { return hwOps.at(name); }
  std::string getInstanceName(Operation *op) const {
    return std::string("instance_") + std::to_string(instanceIds.at(op));
  }

  int64_t getDelay(const std::string& name) const;

  void assignInstanceIds(ModuleOp root);
};

Optional<HWModuleOp> createBodyModule(SPNBody body, ConversionHelper& helper);

ModuleOp convert(ModuleOp root);

class SchedulingProblem : public virtual ::circt::scheduling::Problem {
public:
  SchedulingProblem(Operation *containingOp) {
    setContainingOp(containingOp);
  }  
};

void schedule(ModuleOp root, ConversionHelper& helper, SchedulingProblem& problem);

}