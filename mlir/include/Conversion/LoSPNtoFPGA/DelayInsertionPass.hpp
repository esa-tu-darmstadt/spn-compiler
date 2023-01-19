#pragma once

#include "mlir/Pass/Pass.h"
#include "scheduling.hpp"


namespace mlir::spn::fpga {

class DelayInsertionPass : public PassWrapper<DelayInsertionPass, OperationPass<ModuleOp>> {
  scheduling::SchedulingResult schedule;
public:
  DelayInsertionPass(const scheduling::SchedulingResult& schedule): schedule(schedule) {}
  virtual ~DelayInsertionPass() = default;
  StringRef getArgument() const override { return "insert-delays"; }
  StringRef getDescription() const override { return "Inserts artificial delays to create a functional sequential circuit."; }
  void getDependentDialects(DialectRegistry& registry) const override;
protected:
  void runOnOperation() override;
};

inline std::unique_ptr<mlir::Pass> createDelayInsertionPass(const scheduling::SchedulingResult& schedule) {
  return std::make_unique<DelayInsertionPass>(schedule);
}

}