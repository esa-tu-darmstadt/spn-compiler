#include "LoSPNtoFPGA/LoSPNtoFPGAPass.h"


namespace mlir::spn::fpga {

void LoSPNtoFPGAPass::getDependentDialects(DialectRegistry& registry) const {
  registry.insert<mlir::spn::low::LoSPNDialect>();
  registry.insert<circt::hw::HWDialect>();
  registry.insert<circt::seq::SeqDialect>();
}

void LoSPNtoFPGAPass::runOnOperation() {
  //llvm::errs() << "Not implemented!\n";

  Operation *op = getOperation();
  llvm::outs() << "current op: "; op->dump();

  ModuleOp modOp = llvm::dyn_cast<ModuleOp>(op);

  
}

}