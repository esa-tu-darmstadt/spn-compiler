//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "MLIRtoLLVMIRConversion.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <util/Logging.h>
#include <driver/GlobalOptions.h>

using namespace spnc;

MLIRtoLLVMIRConversion::MLIRtoLLVMIRConversion(spnc::ActionWithOutput<mlir::ModuleOp>& _input,
                                               std::shared_ptr<mlir::MLIRContext> context,
                                               std::shared_ptr<llvm::TargetMachine> targetMachine,
                                               bool optimizeOutput)
    : ActionSingleInput<mlir::ModuleOp, llvm::Module>{_input}, cached{false}, optimize{optimizeOutput},
    ctx{std::move(context)}, machine{std::move(targetMachine)}, llvmCtx{} {}

llvm::Module& spnc::MLIRtoLLVMIRConversion::execute() {
  if (!cached) {
    auto inputModule = input.execute();
    module = mlir::translateModuleToLLVMIR(inputModule, llvmCtx);
    if (spnc::option::dumpIR.get(*this->config)) {
      llvm::dbgs() << "\n// *** IR after conversion to LLVM IR ***\n";
      module->dump();
    }
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
    if (optimize && spnc::option::dumpIR.get(*this->config)) {
      llvm::dbgs() << "\n// *** IR after optimization of LLVM IR ***\n";
      module->dump();
    }
    cached = true;
  }
  return *module;
}