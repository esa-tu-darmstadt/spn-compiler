#pragma once

#include <unordered_map>
#include <string_view>

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
using namespace ::circt::comb;


namespace mlir::spn::fpga::scheduling {

inline const std::string TYPE_ADD = "FPAdd";
inline const std::string TYPE_MUL = "FPMult";
inline const std::string TYPE_LOG = "FPLog";
inline const std::string TYPE_CATEGORICAL = "categorical";
inline const std::string TYPE_HISTOGRAM = "histogram";
inline const std::string TYPE_CONSTANT = "constant";

class DelayLibrary {
  std::unordered_map<std::string, uint64_t> delays;
public:
  DelayLibrary() {
    delays = {
      {TYPE_ADD, 5},
      {TYPE_MUL, 10},
      {TYPE_LOG, 0},
      {TYPE_CATEGORICAL, 1},
      {TYPE_CONSTANT, 0},
      {TYPE_HISTOGRAM, 1}
    };
  }

  uint64_t getDelay(const std::string& operatorType) const {
    return delays.at(operatorType);
  }
};

class SchedulingProblem : public virtual ::circt::scheduling::Problem {
  Operation *root;
  DelayLibrary delayLibrary;
public:
  SchedulingProblem(Operation *containingOp): root(containingOp) {
    setContainingOp(containingOp);
  }

  // Similar to verify() we don't want to do this in the constructor.
  void construct() {
    // insert the operator types and their delays
    for (const auto& type : {TYPE_ADD, TYPE_MUL, TYPE_LOG, TYPE_CATEGORICAL,
                             TYPE_HISTOGRAM, TYPE_CONSTANT}) {
      OperatorType opType = getOrInsertOperatorType(type);
      setLatency(opType, delayLibrary.getDelay(type));
    }

    root->walk([&](Operation *op) {
      std::optional<std::string> optOpType = determineOperatorType(op);

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
        //CompRegOp reg = builder.create<CompRegOp>(
        //  builder.getUnknownLoc(), std::move(prev), clk, "shiftReg"
        //);
        FirRegOp reg = builder.create<FirRegOp>(
          builder.getUnknownLoc(), std::move(prev), clk, builder.getStringAttr("shiftReg")
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

private:
  static std::optional<std::string> determineOperatorType(Operation *op) {
    if (InstanceOp instOp = llvm::dyn_cast<InstanceOp>(op)) {
      Operation *refMod = instOp.getReferencedModule(nullptr);
      assert(refMod);

      // check if the reference is to an external verilog module
      if (HWModuleExternOp extOp = llvm::dyn_cast<HWModuleExternOp>(refMod))
        return extOp.getName().str();

      if (HWModuleOp modOp = llvm::dyn_cast<HWModuleOp>(refMod)) {
        // we need to do some string processing to extract the operator type name
        std::string modName = modOp.getName().str();

        if (modName.find(TYPE_CATEGORICAL) != std::string::npos)
          return TYPE_CATEGORICAL;
        else if (modName.find(TYPE_HISTOGRAM) != std::string::npos)
          return TYPE_HISTOGRAM;
      }

      return std::nullopt;
    } else if (llvm::isa<ConstantOp>(op)) {
      return TYPE_CONSTANT;
    }

    return std::nullopt;
  }
};

}