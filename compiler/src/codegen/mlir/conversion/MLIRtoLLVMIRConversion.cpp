//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "MLIRtoLLVMIRConversion.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <util/Logging.h>
#include <driver/GlobalOptions.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Coroutines.h>
#include "llvm/IR/PassTimingInfo.h"

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

    SPDLOG_INFO("Finished conversion to LLVM IR");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    // NOTE: If we want to support cross-compilation, we need to replace the following line, as it will
    // always set the modules target triple to the native CPU target.
    mlir::ExecutionEngine::setupTargetTriple(module.get());
    // Run optimization pipeline to get rid of some clutter introduced during conversion to LLVM dialect in MLIR.
    optimizeLLVMIR();
    if (optimize && spnc::option::dumpIR.get(*this->config)) {
      llvm::dbgs() << "\n// *** IR after optimization of LLVM IR ***\n";
      module->dump();
    }
    cached = true;
  }
  return *module;
}

void MLIRtoLLVMIRConversion::optimizeLLVMIR() {
  llvm::legacy::PassManager modulePM;
  llvm::legacy::FunctionPassManager funcPM(module.get());
  llvm::PassManagerBuilder builder;
  // TODO Allow more fine-grained setting via option.
  unsigned optLevel = (optimize) ? 3 : 0;
  unsigned sizeLevel = 0;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;
  builder.Inliner = llvm::createFunctionInliningPass(optLevel, sizeLevel, false);
  // Currently both kinds of vectorization are always disabled. Either the
  // vectorization was already performed in MLIR or the user did not request vectorization.
  builder.LoopVectorize = false;
  builder.SLPVectorize = false;
  builder.DisableUnrollLoops = false;

  // Add all coroutine passes to the builder.
  llvm::addCoroutinePassesToExtensionPoints(builder);

  if (machine.get()) {
    // Add pass to initialize TTI for this specific target. Otherwise, TTI will
    // be initialized to NoTTIImpl by default.
    modulePM.add(createTargetTransformInfoWrapperPass(
        machine->getTargetIRAnalysis()));
    funcPM.add(createTargetTransformInfoWrapperPass(
        machine->getTargetIRAnalysis()));
  }

  // Populate the pass managers
  builder.populateModulePassManager(modulePM);
  builder.populateFunctionPassManager(funcPM);

  // Run the pipelines on the module and the contained functions.
  funcPM.doInitialization();
  for (auto& F : *module) {
    funcPM.run(F);
  }
  funcPM.doFinalization();
  modulePM.run(*module);
}