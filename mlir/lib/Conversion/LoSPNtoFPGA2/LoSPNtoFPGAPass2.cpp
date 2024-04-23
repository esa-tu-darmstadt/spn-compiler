#include "LoSPNtoFPGA2/LoSPNtoFPGAPass2.hpp"


namespace mlir::spn::fpga {

void LoSPNtoFPGAPass2::getDependentDialects(DialectRegistry& registry) const {
  //registry.insert<mlir::spn::low::LoSPNDialect>();
  registry.insert<circt::hw::HWDialect>();
  registry.insert<circt::seq::SeqDialect>();
  registry.insert<circt::sv::SVDialect>();
  registry.insert<circt::comb::CombDialect>();
  registry.insert<circt::firrtl::FIRRTLDialect>();
}

void LoSPNtoFPGAPass2::runOnOperation() {
  ModuleOp modOp = getOperation();

  Optional<ModuleOp> newModOp = convert(modOp, options);

  if (!newModOp.has_value() || !succeeded(newModOp.value().verify())) {
    signalPassFailure();
    return;
  }

  // TODO: Why does this not work with a rewriter?
  modOp.getRegion().takeBody(newModOp.value().getRegion());
}

}