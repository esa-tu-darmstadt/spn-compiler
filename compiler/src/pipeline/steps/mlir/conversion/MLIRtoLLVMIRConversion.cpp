//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "MLIRtoLLVMIRConversion.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "option/Options.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/Error.h"
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO.h>
#include <util/Logging.h>

#include "llvm/Passes/PassBuilder.h"

using namespace spnc;

spnc::MLIRtoLLVMIRConversion::MLIRtoLLVMIRConversion(
    StepWithResult<mlir::ModuleOp> &input)
    : StepSingleInput<MLIRtoLLVMIRConversion, mlir::ModuleOp>(input),
      llvmCtx{} {}

ExecutionResult
spnc::MLIRtoLLVMIRConversion::executeStep(mlir::ModuleOp *mlirModule) {
  module = mlir::translateModuleToLLVMIR(mlirModule->getOperation(), llvmCtx);
  auto *config = getContext()->get<Configuration>();
  if (spnc::option::dumpIR.get(*config)) {
    llvm::dbgs() << "\n// *** IR after conversion to LLVM IR ***\n";
#ifdef LLVM_ENABLE_DUMP
    module->dump();
#else
    llvm::dbgs() << "Dumping of LLVM IR is disabled. Consider recompiling LLVM "
                    "with LLVM_ENABLE_DUMP=ON\n";
#endif
  }
  if (!module) {
    return failure("Conversion to LLVM IR failed");
  }

  if (spnc::option::dumpIR) {
    llvm::dbgs() << "\n// *** IR after conversion to LLVM IR ***\n";
    module->dump();
  }

  SPDLOG_INFO("Finished conversion to LLVM IR");

  // Set target triple and data layout
  auto targetMachine = getContext()->get<llvm::TargetMachine>();
  module->setDataLayout(targetMachine->createDataLayout());
  module->setTargetTriple(targetMachine->getTargetTriple().getTriple());

  // Run optimization pipeline to get rid of some clutter introduced during
  // conversion to LLVM dialect in MLIR.
  auto optLevel = retrieveOptLevel();
  optimizeLLVMIR(optLevel);
  if (optLevel > 0 && option::dumpIR) {
    llvm::dbgs() << "\n// *** IR after optimization of LLVM IR ***\n";
#ifdef LLVM_ENABLE_DUMP
    module->dump();
#else
    llvm::dbgs() << "Dumping of LLVM IR is disabled. Consider recompiling LLVM "
                    "with LLVM_ENABLE_DUMP=ON\n";
#endif
  }
  return success();
}

int spnc::MLIRtoLLVMIRConversion::retrieveOptLevel() {
  int irOptLevel = option::irOptLevel.getNumOccurrences() > 0
                       ? option::irOptLevel
                       : option::optLevel;
  if (option::irOptLevel.getNumOccurrences() > 0 &&
      option::irOptLevel != option::optLevel) {
    SPDLOG_INFO("Option ir-opt-level (value: {}) takes precedence over option "
                "opt-level (value: {})",
                option::irOptLevel, option::optLevel);
  }
  return irOptLevel;
}

void MLIRtoLLVMIRConversion::optimizeLLVMIR(int irOptLevel) {
  unsigned sizeLevel = 0;
  auto machine = getContext()->get<llvm::TargetMachine>();
  auto optPipeline =
      mlir::makeOptimizingTransformer(irOptLevel, sizeLevel, machine);

  auto optPipelineCustom = [irOptLevel, sizeLevel,
                            machine](llvm::Module *m) -> llvm::Error {
    llvm::OptimizationLevel ol;

    switch (irOptLevel) {
    case 0:
      ol = llvm::OptimizationLevel::O0;
      break;

    case 1:
      ol = llvm::OptimizationLevel::O1;
      break;

    case 2:
      switch (sizeLevel) {
      case 0:
        ol = llvm::OptimizationLevel::O2;
        break;

      case 1:
        ol = llvm::OptimizationLevel::Os;
        break;

      case 2:
        ol = llvm::OptimizationLevel::Oz;
      }
      break;
    case 3:
      ol = llvm::OptimizationLevel::O3;
      break;
    }

    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = true;
    tuningOptions.SLPVectorization = false;

    llvm::PassBuilder pb(machine, tuningOptions);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm;
    mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
    mpm.run(*m, mam);
    return llvm::Error::success();
  };

  if (auto err = optPipelineCustom(module.get())) {
    SPDLOG_ERROR("Failed to optimize LLVM IR");
    llvm::report_fatal_error(std::move(err));
  } else {
    SPDLOG_DEBUG("Finished optimization of LLVM IR");
  }
}

llvm::Module *MLIRtoLLVMIRConversion::result() { return module.get(); }