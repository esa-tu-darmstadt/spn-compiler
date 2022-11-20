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
#include <option/GlobalOptions.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Coroutines.h>
#include "llvm/IR/PassTimingInfo.h"

using namespace spnc;

spnc::MLIRtoLLVMIRConversion::MLIRtoLLVMIRConversion(StepWithResult<mlir::ModuleOp>& input) : StepSingleInput<
    MLIRtoLLVMIRConversion,
    mlir::ModuleOp>(input), llvmCtx{} {}

ExecutionResult spnc::MLIRtoLLVMIRConversion::executeStep(mlir::ModuleOp* mlirModule) {
  module = mlir::translateModuleToLLVMIR(mlirModule->getOperation(), llvmCtx);
  auto* config = getContext()->get<Configuration>();
  if (spnc::option::dumpIR.get(*config)) {
    llvm::dbgs() << "\n// *** IR after conversion to LLVM IR ***\n";
    module->dump();
  }
  if (!module) {
    return failure("Conversion to LLVM IR failed");
  }

  SPDLOG_INFO("Finished conversion to LLVM IR");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  // NOTE: If we want to support cross-compilation, we need to replace the following line, as it will
  // always set the modules target triple to the native CPU target.
  mlir::ExecutionEngine::setupTargetTriple(module.get());
  // Run optimization pipeline to get rid of some clutter introduced during conversion to LLVM dialect in MLIR.
  auto optLevel = retrieveOptLevel();
  optimizeLLVMIR(optLevel);
  if (optLevel > 0 && spnc::option::dumpIR.get(*config)) {
    llvm::dbgs() << "\n// *** IR after optimization of LLVM IR ***\n";
    module->dump();
  }
  return success();
}

int spnc::MLIRtoLLVMIRConversion::retrieveOptLevel() {
  auto* config = getContext()->get<Configuration>();
  int irOptLevel = spnc::option::optLevel.get(*config);
  if (spnc::option::irOptLevel.isPresent(*config) && spnc::option::irOptLevel.get(*config) != irOptLevel) {
    auto optionValue = spnc::option::irOptLevel.get(*config);
    SPDLOG_INFO("Option ir-opt-level (value: {}) takes precedence over option opt-level (value: {})",
                optionValue, irOptLevel);
    irOptLevel = optionValue;
  }
  return irOptLevel;
}

void MLIRtoLLVMIRConversion::optimizeLLVMIR(int irOptLevel) {
  llvm::legacy::PassManager modulePM;
  llvm::legacy::FunctionPassManager funcPM(module.get());
  llvm::PassManagerBuilder builder;
  unsigned sizeLevel = 0;
  builder.OptLevel = irOptLevel;
  builder.SizeLevel = sizeLevel;
  builder.Inliner = llvm::createFunctionInliningPass(irOptLevel, sizeLevel, false);
  // Currently both kinds of vectorization are always disabled. Either the
  // vectorization was already performed in MLIR or the user did not request vectorization.
  builder.LoopVectorize = false;
  builder.SLPVectorize = false;
  builder.DisableUnrollLoops = false;

  // Add all coroutine passes to the builder.
  // TODO: What about this?
  //llvm::addCoroutinePassesToExtensionPoints(builder);

  auto machine = getContext()->get<llvm::TargetMachine>();
  if (machine) {
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
  for (auto& F: *module) {
    funcPM.run(F);
  }
  funcPM.doFinalization();
  modulePM.run(*module);
}

llvm::Module* MLIRtoLLVMIRConversion::result() {
  return module.get();
}