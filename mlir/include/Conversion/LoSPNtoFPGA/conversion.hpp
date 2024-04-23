#pragma once

#include <unordered_map>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/HWArith/HWArithTypes.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"

#include "HiSPN/HiSPNDialect.h"
#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"

#include "scheduling.hpp"
#include "types.hpp"

using namespace ::mlir;
using namespace ::circt::hw;
using namespace ::mlir::spn::low;
using namespace ::mlir::spn::high;
using namespace ::circt::seq;
using namespace ::circt::sv;
using namespace ::circt::comb;
using namespace ::mlir::spn::fpga::operators;
using namespace ::mlir::spn::fpga::types;

namespace mlir::spn::fpga {

class ConversionHelper {
  MLIRContext *ctxt;
  OpBuilder builder;

  Type indexType, probType, sigType;

  std::unordered_map<Operation *, uint64_t> instanceIds;
  // contains external ops
  std::unordered_map<OperatorType, Operation *> hwOps;
  // contains all categoricals and histograms
  std::unordered_map<Operation *, Operation *> leafModules;

  std::vector<PortInfo> hwPorts(const std::vector<OperatorPortInfo> &ports);
  void createHwOps();

public:
  OperatorTypeMapping opMapping;
  TargetTypes targetTypes;

  ConversionHelper(MLIRContext *ctxt)
      : ctxt(ctxt), builder(ctxt), targetTypes(builder) {
    indexType = builder.getI8Type();
    probType = builder.getIntegerType(31);
    sigType = builder.getI1Type();

    createHwOps();
  }

  MLIRContext *getContext() const { return ctxt; }
  OpBuilder getBuilder() const { return builder; }
  // Type getIndexType() const { return indexType; }
  // Type getProbType() const { return probType; }
  // Type getSigType() const { return sigType; }

  // PortInfo port(const OperatorPortInfo& portInfo) {
  //   return PortInfo{
  //     .name = builder.getStringAttr(portInfo.name),
  //     .direction = portInfo.direction,
  //     .type = portInfo.type
  //   };
  // }

  PortInfo port(const std::string &name, PortDirection direction, Type type) {
    return PortInfo{.name = builder.getStringAttr(name),
                    .direction = direction,
                    .type = type};
  };

  PortInfo inPort(const std::string &name, Type type) {
    return port(name, PortDirection::INPUT, type);
  }
  PortInfo outPort(const std::string &name, Type type) {
    return port(name, PortDirection::OUTPUT, type);
  }

  Operation *getMod(OperatorType type) const { return hwOps.at(type); }
  std::string getInstanceName(Operation *op) const {
    return std::string("instance_") + std::to_string(instanceIds.at(op));
  }
  uint64_t getInstanceId(Operation *op) const { return instanceIds.at(op); }

  int64_t getDelay(const std::string &name) const;

  void assignInstanceIds(ModuleOp root);
  void assignLeafModules(ModuleOp root);
  std::unordered_map<Operation *, Operation *> &getLeafModules() {
    return leafModules;
  }
  Operation *getLeafModule(Operation *op) const { return leafModules.at(op); }

  std::string getVerilogIncludeString() const {
    return "";
    return R"(
`include "FPAdd.v"
`include "FPMult.v"
`include "FPLog.v"
    )";
  }

private:
  Optional<HWModuleOp> createLeafModule(Operation *op);
  std::vector<Value> createProbabilityArray(OpBuilder &builder, Operation *op);
};

Optional<HWModuleOp> createBodyModule(SPNBody body, ConversionHelper &helper);

Optional<ModuleOp> convert(ModuleOp root);

class SchedulingProblem : public virtual ::circt::scheduling::Problem {
public:
  SchedulingProblem(Operation *containingOp) { setContainingOp(containingOp); }
};

void schedule(HWModuleOp root, ConversionHelper &helper,
              SchedulingProblem &problem);

void insertShiftRegisters(HWModuleOp root, ConversionHelper &helper,
                          SchedulingProblem &problem);

} // namespace mlir::spn::fpga