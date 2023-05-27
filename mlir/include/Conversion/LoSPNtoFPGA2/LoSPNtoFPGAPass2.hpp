#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/PatternMatch.h"

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

#include "circt/Dialect/HW/HWDialect.h"

#include "circt/Dialect/Seq/SeqDialect.h"


namespace mlir::spn::fpga {

// OperationPass<ModuleOp> guarantees that getOperation() always returns a ModuleOp!
struct LoSPNtoFPGAPass2 : public PassWrapper<LoSPNtoFPGAPass2, OperationPass<ModuleOp>> {
public:
  LoSPNtoFPGAPass2() = default;
  virtual ~LoSPNtoFPGAPass2() = default;
  StringRef getArgument() const override { return "convert-lospn-to-fpga-2"; }
  StringRef getDescription() const override { return "Converts a SPN in LoSPN format to a format that can be exported to verilog using circt-opt."; }
  void getDependentDialects(DialectRegistry& registry) const override;
protected:
  void runOnOperation() override;
};

inline std::unique_ptr<mlir::Pass> createLoSPNtoFPGAPass2() {
  return std::make_unique<LoSPNtoFPGAPass2>();
}

}