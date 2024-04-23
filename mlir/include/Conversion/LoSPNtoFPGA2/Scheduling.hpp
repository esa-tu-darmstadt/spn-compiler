#pragma once

#include "HiSPN/HiSPNDialect.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"

namespace mlir::spn::fpga {

using namespace low;

class SchedulingProblem : public virtual ::circt::scheduling::Problem {
  Operation *root;
  DenseSet<Operation *> whitelist;

public:
  SchedulingProblem(SPNBody body) : root(body.getOperation()) {
    setContainingOp(body.getOperation());
  }

  void
  construct(const std::function<std::tuple<uint32_t, std::string>(Operation *)>
                &getDelayAndType);
  uint32_t getDelay(Operation *parent, Operation *child);
  uint32_t getDelay(Value operand, Operation *op);
  uint32_t getTotalEndTime();
};

} // namespace mlir::spn::fpga