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
                                               std::shared_ptr<mlir::MLIRContext> context,
                                               std::shared_ptr<llvm::TargetMachine> targetMachine,
                                               bool optimizeOutput)
    : ActionSingleInput<mlir::ModuleOp, llvm::Module>{_input}, optimize{optimizeOutput},
      ctx{std::move(context)}, machine{std::move(targetMachine)}, llvmCtx{} {}

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
    // NOTE: If we want to support cross-compilation, we need to replace the following line, as it will
    // always set the modules target triple to the native CPU target.
    mlir::ExecutionEngine::setupTargetTriple(module.get());
    // Run optimization pipeline to get rid of some clutter introduced during conversion to LLVM dialect in MLIR.
    auto optPipeline = mlir::makeOptimizingTransformer((optimize ? 3 : 0), 0, machine.get());
    if (auto err = optPipeline(module.get())) {
      SPNC_FATAL_ERROR("Optimization of converted LLVM IR failed");
    }
    module->dump();
    cached = true;
  }
  return *module;
}