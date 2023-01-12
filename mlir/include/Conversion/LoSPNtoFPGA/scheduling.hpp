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

// We want to further detach the operator type string from the verilog
// module name because the latter should be freely configurable by the user.
// For that the user must implement the methods in DelayLibrary to their needs.
enum OpType {
  // these come from external verilog sources
  TYPE_ADD, TYPE_MUL, TYPE_LOG,
  // these are generated
  TYPE_CATEGORICAL, TYPE_HISTOGRAM, TYPE_CONSTANT
};

class DelayLibrary {
protected:
  std::unordered_map<OpType, uint64_t> delays;
  std::unordered_map<std::string, uint64_t> delaysByName;

  // contains for example FPAdd -> TYPE_ADD
  std::unordered_map<std::string, OpType> externalNameToType;
public:
  DelayLibrary() {
    // TODO: Load this from a config file or similar.

    // for now these are the delays for the ufloat operators with 8;23 bits
    // FPOps_build_mult delay: 5
    // FPOps_build_add delay: 6
    delays = {
      {TYPE_ADD, 6},
      {TYPE_MUL, 5},
      {TYPE_LOG, 0},
      {TYPE_CATEGORICAL, 1},
      {TYPE_CONSTANT, 0},
      {TYPE_HISTOGRAM, 1}
    };

    for (const auto& pair : delays)
      delaysByName[typeToString(std::get<0>(pair))] = std::get<1>(pair);

    externalNameToType = {
       {"FPAdd", TYPE_ADD},
       {"FPMult", TYPE_MUL},
       {"FPLog", TYPE_LOG}
    };
  } 
protected:
  std::string typeToString(OpType type) const {
    switch (type) {
      case TYPE_ADD: return "TYPE_ADD";
      case TYPE_MUL: return "TYPE_MUL";
      case TYPE_LOG: return "TYPE_LOG";
      case TYPE_CATEGORICAL: return "TYPE_CATEGORICAL";
      case TYPE_HISTOGRAM: return "TYPE_HISTOGRAM";
      case TYPE_CONSTANT: return "TYPE_CONSTANT";
    }

    return "TYPE_UNKNOWN";
  }

  // The following 2 functions basically encode the naming conventions used
  // to identify the modules.
  std::optional<OpType> getExternalType(const std::string& name) const {
    return externalNameToType.at(name);
  }

  virtual std::optional<OpType> getGeneratedModuleType(const std::string& name) const {
    if (name.find("categorical") != std::string::npos)
      return TYPE_CATEGORICAL;
    else if (name.find("histogram") != std::string::npos)
      return TYPE_HISTOGRAM;
    else
      return std::nullopt;
  }

  std::optional<OpType> _determineOperatorType(Operation *op) const {
    if (InstanceOp instOp = llvm::dyn_cast<InstanceOp>(op)) {
      Operation *refMod = instOp.getReferencedModule(nullptr);
      assert(refMod);

      // check if the reference is to an external verilog module
      if (HWModuleExternOp extOp = llvm::dyn_cast<HWModuleExternOp>(refMod)) {
        auto it = externalNameToType.find(extOp.getName().str());

        if (it != externalNameToType.end())
          return std::get<1>(*it);
      } else if (HWModuleOp modOp = llvm::dyn_cast<HWModuleOp>(refMod)) {
        // we need to do some string processing to extract the operator type name
        return getGeneratedModuleType(modOp.getName().str());
      }
    } else if (llvm::isa<ConstantOp>(op)) {
      return TYPE_CONSTANT;
    }

    return std::nullopt;
  }
public:
  std::optional<std::string> determineOperatorType(Operation *op) const {
    std::optional<OpType> opType = _determineOperatorType(op);

    if (!opType)
      return std::nullopt;

    return typeToString(opType.value());
  }

  void insertDelays(::circt::scheduling::Problem& problem) {
    for (const auto& pair : delaysByName) {
      auto opType = problem.getOrInsertOperatorType(std::get<0>(pair));
      problem.setLatency(opType, std::get<1>(pair));
    }
  }
};

class SchedulingProblem : public virtual ::circt::scheduling::Problem {
  Operation *root;
  DelayLibrary delayLibrary;
public:
  SchedulingProblem(Operation *containingOp): root(containingOp) {
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
};

}