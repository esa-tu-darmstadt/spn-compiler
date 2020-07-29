//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNToLLVMLowering.h"
#include <mlir/Target/LLVMIR.h>
#include <codegen/mlir/transform/passes/SPNMLIRPasses.h>
#include <codegen/mlir/lowering/passes/SPNLoweringPasses.h>
#include <util/Logging.h>

using namespace mlir;
using namespace spnc;

SPNToLLVMLowering::SPNToLLVMLowering(spnc::ActionWithOutput<ModuleOp>& _input, std::shared_ptr<MLIRContext> _mlirContext)
    : ActionSingleInput<ModuleOp, ModuleOp>{_input}, mlirContext{std::move(_mlirContext)}, pm{mlirContext.get()} {
  pm.addPass(mlir::spn::createSPNtoLLVMLoweringPass());
}

ModuleOp& SPNToLLVMLowering::execute() {
  if (!cached) {
    auto inputModule = input.execute();
    // Clone the module to keep the original module available
    // for actions using the same input module.
    module = std::make_unique<ModuleOp>(inputModule.clone());
    auto result = pm.run(*module);
    if (failed(result)) {
      SPNC_FATAL_ERROR("Running the MLIR pass 'SPN to LLVM lowering' failed!");
    }
    cached = true;
  }
  return *module;
}
