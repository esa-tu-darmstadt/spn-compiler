#include "CIRCTToolchain.h"


namespace spnc {

void CIRCTToolchain::initializeMLIRContext(mlir::MLIRContext *ctx) {
  // TODO
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::spn::high::HiSPNDialect>();
  registry.insert<mlir::spn::low::LoSPNDialect>();
  ctx.loadDialect<mlir::spn::high::HiSPNDialect>();
  ctx.loadDialect<mlir::spn::low::LoSPNDialect>();
  ctx.appendDialectRegistry(registry);
  mlir::registerLLVMDialectTranslation(ctx);
  for (auto* D : ctx.getLoadedDialects()) {
    SPDLOG_INFO("Loaded dialect: {}", D->getNamespace().str());
  }
}

}