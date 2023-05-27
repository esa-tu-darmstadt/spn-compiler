#include "LoSPNtoFPGA2/Scheduling.hpp"


namespace mlir::spn::fpga {

void SchedulingProblem::construct(const std::function<std::tuple<uint32_t, std::string>(Operation *)>& getDelayAndType) {
  root->walk([&](Operation *op) {
    if (isa<SPNBody>(op))
      return;

    auto [delay, type] = getDelayAndType(op);

    // we don't care about the delay of constants
    if (type == "const")
      whitelist.insert(op);

    // tell the problem what we are dealing with
    auto opType = getOrInsertOperatorType(type);
    setLatency(opType, delay);
    setLinkedOperatorType(op, opType);
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

uint32_t SchedulingProblem::getDelay(Operation *parent, Operation *child) {
  uint32_t t1 = getEndTime(parent).value();
  uint32_t t2 = getStartTime(child).value();
  assert(t2 >= t1 && "child starts before parent ends");
  return t2 - t1;
}

uint32_t SchedulingProblem::getDelay(Value operand, Operation *op) {
  if (!operand.getDefiningOp() || whitelist.contains(operand.getDefiningOp()))
    return 0;

  return getDelay(operand.getDefiningOp(), op);
}

}