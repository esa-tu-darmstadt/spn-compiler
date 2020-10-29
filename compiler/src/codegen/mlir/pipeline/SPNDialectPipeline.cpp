//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNDialectPipeline.h"
#include "mlir/Transforms/Passes.h"
#include "SPN/SPNPasses.h"
#include "SPNtoStandard/SPNtoStandardConversionPass.h"
#include "SPNtoLLVM/SPNtoLLVMConversionPass.h"
#include <util/Logging.h>

using namespace mlir;
using namespace spnc;

SPNDialectPipeline::SPNDialectPipeline(ActionWithOutput<mlir::ModuleOp>& _input,
                                       std::shared_ptr<mlir::MLIRContext> _mlirContext)
    : ActionSingleInput<ModuleOp, ModuleOp>{_input},
      mlirContext{std::move(_mlirContext)}, pm{mlirContext.get()} {
  pm.addPass(mlir::spn::createSPNOpSimplifierPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::spn::createSPNTypePinningPass());
  pm.addPass(mlir::spn::createSPNtoStandardConversionPass());
  pm.addPass(mlir::spn::createSPNtoLLVMConversionPass());
}

ModuleOp& SPNDialectPipeline::execute() {
  if (!cached) {
    auto inputModule = input.execute();
    // Clone the module to keep the original module available
    // for actions using the same input module.
    module = std::make_unique<ModuleOp>(inputModule.clone());
    auto result = pm.run(*module);
    if (failed(result)) {
      SPNC_FATAL_ERROR("Running the MLIR pass pipeline failed!");
    }
    cached = true;
  }
  module->dump();
  return *module;
}