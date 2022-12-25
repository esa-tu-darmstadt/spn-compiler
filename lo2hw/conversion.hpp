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


using namespace ::mlir;
using namespace ::circt::hw;
using namespace ::mlir::spn::low;
using namespace ::mlir::spn::high;
using namespace ::circt::seq;


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

void insertShiftRegisters(ModuleOp root, ConversionHelper& helper, SchedulingProblem& problem);

class ShiftRegisterPattern : public OpConversionPattern<InstanceOp> {
  ConversionHelper& helper;
  SchedulingProblem& problem;

  Value delaySignal(ConversionPatternRewriter &rewriter, Value input, Value clk, Value rst, Value rstValue, uint32_t delay) const {
    Value prev = input;

    for (uint32_t i = 0; i < delay; ++i) {
      prev = rewriter.create<CompRegOp>(
        rewriter.getUnknownLoc(),
        prev,
        clk,
        rst,
        rstValue,
        "lalalala"
      ).getResult();
    }

    return prev;
  }
public:
  ShiftRegisterPattern(ConversionHelper& helper, SchedulingProblem& problem, MLIRContext *ctxt):
    OpConversionPattern<InstanceOp>::OpConversionPattern(ctxt),
    helper(helper), problem(problem) {}

  LogicalResult matchAndRewrite(InstanceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    std::vector<Value> delayedOperands;

    Value clk = adaptor.getOperands()[0];
    Value rst = adaptor.getOperands()[1];
    Value rstValue = Value(); // TODO

    for (uint32_t i = 0; i < adaptor.getOperands().size(); ++i) {
      Value operand = adaptor.getOperands()[i];

      // ignore clk, rst
      if (i <= 1) {
        delayedOperands.push_back(operand);
        continue;
      }

      Operation *defOp = operand.getDefiningOp();

      // does not stem from an operation and does therefore not need to be delay
      if (!defOp) {
        delayedOperands.push_back(operand);
        continue;
      }

      Optional<unsigned> startTime = problem.getStartTime(defOp);
      Optional<unsigned> latency = problem.getLatency(problem.getLinkedOperatorType(defOp).value());
      Optional<unsigned> meStartTime = problem.getStartTime(op);

      if (!startTime.has_value() || !latency.has_value() || !meStartTime.has_value())
        return failure();

      uint32_t operandFinishTime = startTime.value() + latency.value();

      if (operandFinishTime > meStartTime.value())
        return failure();

      uint32_t artificialDelay = meStartTime.value() - operandFinishTime;
      Value delayedOperand = delaySignal(rewriter, operand, clk, rst, rstValue, artificialDelay);

      delayedOperands.push_back(delayedOperand);
    }

    rewriter.replaceOpWithNewOp<InstanceOp>(op,
      op.getReferencedModule(),
      op.getName(),
      ArrayRef<Value>(delayedOperands)
    );

    return success();
  }
};

void test(MLIRContext *ctxt);

}