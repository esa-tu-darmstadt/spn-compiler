#include "LoSPNtoFPGA2/LoSPNtoFPGAPass2.hpp"

//#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "LoSPNtoFPGA2/Conversion.hpp"


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

  ConversionOptions options;
  options.ufloatConfig.exponentWidth = 8;
  options.ufloatConfig.mantissaWidth = 23;
  options.use32Bit = true;
  options.performLowering = false;

  Optional<ModuleOp> newModOp = convert(modOp, options);

  if (!newModOp.has_value() || !succeeded(newModOp.value().verify())) {
    signalPassFailure();
    return;
  }

  // TODO: Why does this not work with a rewriter?
  modOp.getRegion().takeBody(newModOp.value().getRegion());
}

}