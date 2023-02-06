#pragma once

#include <unordered_map>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "HiSPN/HiSPNDialect.h"

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"

#include "util/json.hpp"

#include "operators.hpp"


using namespace ::mlir;
using namespace ::circt::hw;
using namespace ::mlir::spn::low;
using namespace ::mlir::spn::high;
using namespace ::circt::seq;
using namespace ::mlir::spn::fpga::operators;


namespace mlir::spn::fpga::scheduling {

class DelayLibrary {
  const OperatorTypeMapping& opMapping;
public:
  DelayLibrary(const OperatorTypeMapping& opMapping): opMapping(opMapping) {}

  std::optional<std::string> determineOperatorType(Operation *op) const {
    if (!opMapping.isMapped(op))
      return std::nullopt;

    return opMapping.getTypeBaseName(opMapping.getType(op));
  }

  void insertDelays(::circt::scheduling::Problem& problem) {
    for (OperatorType opType : {TYPE_ADD, TYPE_MUL, TYPE_LOG, TYPE_CATEGORICAL, TYPE_HISTOGRAM, TYPE_CONSTANT}) {
      auto type = problem.getOrInsertOperatorType(opMapping.getTypeBaseName(opType));
      problem.setLatency(type, opMapping.getDelay(opType));
    }
  }

  uint64_t getDelay(Operation *op) const {
    if (!opMapping.isMapped(op))
      return 0;

    return opMapping.getDelay(opMapping.getType(op));
  }
};

struct SchedulingResult {
  std::vector<std::tuple<
      Operation *,  // from
      Operation *,  // to
      uint32_t      // delay
    >> delays;
  uint32_t totalDelay;
};

class SchedulingProblem : public virtual ::circt::scheduling::Problem {
  Operation *root;
  DelayLibrary delayLibrary;
  SchedulingResult result;
public:
  SchedulingProblem(Operation *containingOp, const OperatorTypeMapping& opMapping): root(containingOp), delayLibrary(opMapping) {
    setContainingOp(containingOp);
  }

  // Similar to check() we don't want to do this in the constructor.
  void construct() {
    // insert the operator types and their delays
    delayLibrary.insertDelays(*this);

    root->walk([&](Operation *op) {
      std::optional<std::string> optOpType = delayLibrary.determineOperatorType(op);

      if (!optOpType)
        return;

      // tell the problem what we are dealing with
      std::string opType = optOpType.value();
      setLinkedOperatorType(op, getOrInsertOperatorType(opType));
      insertOperation(op);

      // add dependencies
      for (Value operand : op->getOperands()) {
        if (!operand.getDefiningOp())
          continue;
        
        assert(succeeded(
          insertDependence(std::make_pair(operand.getDefiningOp(), op))
        ));
      }
    });
  }

  void insertDelays() {
    OpBuilder builder(root->getContext());

    std::vector<std::tuple<
      Operation *,  // from
      Operation *,  // to
      uint32_t      // delay
    >> jobs;

    uint32_t maxEndTime = 0;

    // first collect all the delays we want to insert
    root->walk([&](InstanceOp op) {
      for (Value operand : op.getOperands()) {
        Operation *defOp = operand.getDefiningOp();

        // is a block argument or something
        if (!defOp)
          continue;

        assert(getStartTime(op).has_value());
        assert(getLatency(getLinkedOperatorType(op).value()).has_value());
        assert(getStartTime(defOp).has_value());

        uint32_t meStartTime = getStartTime(op).value();
        uint32_t defOpLatency = getLatency(getLinkedOperatorType(defOp).value()).value();
        uint32_t defOpStartTime = getStartTime(defOp).value();

        assert(defOpStartTime + defOpLatency <= meStartTime);

        uint32_t delay = meStartTime - (defOpStartTime + defOpLatency);

        uint32_t meDelay = getLatency(getLinkedOperatorType(op).value()).value();
        maxEndTime = std::max(maxEndTime, meStartTime + meDelay);

        if (delay == 0)
          continue;

        jobs.emplace_back(
          defOp,
          op.getOperation(),
          delay
        );
      }
    });

    result.totalDelay = maxEndTime;

    // add discardable attributes containg information about the scheduling
    root->walk([&](Operation *op) {
      if (!llvm::isa<InstanceOp>(op) && !llvm::isa<ConstantOp>(op))
        return;

      uint32_t startTime = getStartTime(op).value();
      uint32_t delay = getLatency(getLinkedOperatorType(op).value()).value();

      op->setAttr("fpga.startTime", builder.getI32IntegerAttr(startTime));
      op->setAttr("fpga.delay", builder.getI32IntegerAttr(delay));
    });

    Value clk, rst;

    // stupid way to find the clk and rst signals
    root->walk([&](HWModuleOp op) {
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
        Value rstValue = builder.create<ConstantOp>(
          builder.getUnknownLoc(),
          input.getType(),
          0
        ).getResult();

        FirRegOp reg = builder.create<FirRegOp>(
          builder.getUnknownLoc(), std::move(prev), clk, builder.getStringAttr("shiftReg"),
          rst, rstValue
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

    /*
    
    OpBuilder builder(root->getContext());
    uint32_t maxEndTime = 0;

    // first collect all the delays we want to insert
    root->walk([&](InstanceOp op) {
      for (Value operand : op.getOperands()) {
        Operation *defOp = operand.getDefiningOp();

        // is a block argument or something
        if (!defOp)
          continue;

        assert(getStartTime(op).has_value());
        assert(getLatency(getLinkedOperatorType(op).value()).has_value());
        assert(getStartTime(defOp).has_value());

        uint32_t meStartTime = getStartTime(op).value();
        uint32_t meLatency = getLatency(getLinkedOperatorType(op).value()).value();
        maxEndTime = std::max(maxEndTime, meStartTime + meLatency);

        uint32_t defOpLatency = getLatency(getLinkedOperatorType(defOp).value()).value();
        uint32_t defOpStartTime = getStartTime(defOp).value();

        assert(defOpStartTime + defOpLatency <= meStartTime);

        uint32_t delay = meStartTime - (defOpStartTime + defOpLatency);

        if (delay == 0)
          continue;

        result.delays.emplace_back(
          defOp,
          op.getOperation(),
          delay
        );
      }
    });

    result.totalDelay = maxEndTime;
    
    */

  }

  SchedulingResult getResult() const { return result; }

  void writeSchedule(llvm::raw_fd_ostream& os) {
    using json = nlohmann::json;

    json js;
    js["endTime"] = getResult().totalDelay;

    os << js.dump() << "\n";
  }

  void insertScheduleAsAttribute() {
    OpBuilder builder(root->getContext());
    root->setAttr("fpga.body_delay", builder.getI32IntegerAttr(getResult().totalDelay));
  }
};

}