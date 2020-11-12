//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "MLIRtoLLVMIRConversion.h"
#include <mlir/Target/LLVMIR.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <util/Logging.h>

using namespace spnc;

MLIRtoLLVMIRConversion::MLIRtoLLVMIRConversion(spnc::ActionWithOutput<mlir::ModuleOp>& _input,
                                               std::shared_ptr<mlir::MLIRContext> context, bool optimizeOutput)
    : ActionSingleInput<mlir::ModuleOp, llvm::Module>{_input}, optimize{optimizeOutput},
      ctx{std::move(context)}, llvmCtx{} {}

llvm::Module& spnc::MLIRtoLLVMIRConversion::execute() {
  if (!cached) {
    auto inputModule = input.execute();
    inputModule.dump();
    module = mlir::translateModuleToLLVMIR(inputModule, llvmCtx);
    if (!module) {
      SPNC_FATAL_ERROR("Conversion to LLVM IR failed");
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(module.get());
    // Run optimization pipeline to get rid of some clutter introduced during conversion to LLVM dialect in MLIR.
    auto optPipeline = mlir::makeOptimizingTransformer((optimize ? 3 : 0), 0, nullptr);
    if (auto err = optPipeline(module.get())) {
      SPNC_FATAL_ERROR("Optimization of converted LLVM IR failed");
    }
    module->dump();
    cached = true;
  }
  return *module;
}