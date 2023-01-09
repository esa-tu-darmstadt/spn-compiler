#include "LoSPNtoFPGA/LoSPNtoFPGAPass.h"

#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "LoSPNtoFPGA/conversion.hpp"


namespace mlir::spn::fpga {

void LoSPNtoFPGAPass::getDependentDialects(DialectRegistry& registry) const {
  //registry.insert<mlir::spn::low::LoSPNDialect>();
  registry.insert<circt::hw::HWDialect>();
  registry.insert<circt::seq::SeqDialect>();
  registry.insert<circt::sv::SVDialect>();
}

void LoSPNtoFPGAPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  Optional<ModuleOp> newModOp = convert(modOp);

  if (!newModOp.has_value() || !succeeded(newModOp.value().verify())) {
    signalPassFailure();
    return;
  }

  // TODO: Why does this not work with a rewriter?
  modOp.getRegion().takeBody(newModOp.value().getRegion());
}

}