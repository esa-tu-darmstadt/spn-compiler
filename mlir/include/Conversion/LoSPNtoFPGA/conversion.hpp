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

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"


using namespace ::mlir;
using namespace ::circt::hw;
using namespace ::mlir::spn::low;
using namespace ::mlir::spn::high;
using namespace ::circt::seq;
using namespace ::circt::sv;


namespace mlir::spn::fpga {

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
  OpBuilder getBuilder() const { return builder; }
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
  uint64_t getInstanceId(Operation *op) const { return instanceIds.at(op); }

  int64_t getDelay(const std::string& name) const;

  void assignInstanceIds(ModuleOp root);

  std::string getVerilogIncludeString() const {
    return R"(
`include "FPAdd.sv"
`include "FPMult.sv"
    )";
  }

  Value getProbConstant(double value) {
    int64_t bits = *reinterpret_cast<const int64_t *>(&value);

    return builder.create<ConstantOp>(
      builder.getUnknownLoc(),
      getProbType(),
      123456 //bits
    ).getResult();
  }
};

Optional<HWModuleOp> createBodyModule(SPNBody body, ConversionHelper& helper);

Optional<ModuleOp> convert(ModuleOp root);

class SchedulingProblem : public virtual ::circt::scheduling::Problem {
public:
  SchedulingProblem(Operation *containingOp) {
    setContainingOp(containingOp);
  }  
};

void schedule(ModuleOp root, ConversionHelper& helper, SchedulingProblem& problem);

void insertShiftRegisters(ModuleOp root, ConversionHelper& helper, SchedulingProblem& problem);

Optional<HWModuleOp> createCategoricalModule(ConversionHelper& helper, SPNCategoricalLeaf op, uint64_t id);

}