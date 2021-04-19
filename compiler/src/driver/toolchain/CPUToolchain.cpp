//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "CPUToolchain.h"
#include <driver/BaseActions.h>
#include "codegen/mlir/conversion/HiSPNtoLoSPNConversion.h"
#include "codegen/mlir/conversion/LoSPNtoCPUConversion.h"
#include "codegen/mlir/conversion/CPUtoLLVMConversion.h"
#include "codegen/mlir/conversion/MLIRtoLLVMIRConversion.h"
#include "codegen/mlir/analysis/CollectGraphStatistics.h"
#include <driver/action/ClangKernelLinking.h>
#include <codegen/mlir/frontend/MLIRDeserializer.h>
#include <codegen/mlir/transformation/LoSPNTransformations.h>
#include <driver/action/EmitObjectCode.h>

using namespace spnc;
using namespace mlir;

std::unique_ptr<Job<Kernel> > CPUToolchain::constructJobFromFile(const std::string& inputFile,
                                                                 std::shared_ptr<interface::Configuration> config) {
  // Uncomment the following two lines to get detailed output during MLIR dialect conversion;
  //llvm::DebugFlag = true;
  //llvm::setCurrentDebugType("dialect-conversion");
  std::unique_ptr<Job<Kernel>> job = std::make_unique<Job<Kernel>>(config);
  // Invoke MLIR code-generation on parsed tree.
  auto ctx = std::make_shared<MLIRContext>();
  initializeMLIRContext(*ctx);
  // If IR should be dumped between steps/passes, we need to disable
  // multi-threading in MLIR
  if (spnc::option::dumpIR.get(*config)) {
    ctx->enableMultithreading(false);
  }
  auto diagHandler = setupDiagnosticHandler(ctx.get());
  auto cpuVectorize = spnc::option::cpuVectorize.get(*config);
  SPDLOG_INFO("CPU Vectorization enabled: {}", cpuVectorize);
  auto targetMachine = createTargetMachine(cpuVectorize);
  auto kernelInfo = std::make_shared<KernelInfo>();
  kernelInfo->target = KernelTarget::CPU;
  BinarySPN binarySPNFile{inputFile, false};
  auto& deserialized = job->insertAction<MLIRDeserializer>(std::move(binarySPNFile), ctx, kernelInfo);
  auto& hispn2lospn = job->insertAction<HiSPNtoLoSPNConversion>(deserialized, ctx, diagHandler);
  auto& lospnTransform = job->insertAction<LoSPNTransformations>(hispn2lospn, ctx, diagHandler, kernelInfo);
  ActionWithOutput<ModuleOp>* lospnTransformResult = &lospnTransform;
  // If requested via the configuration, collect graph statistics.
  if (spnc::option::collectGraphStats.get(*config)) {
    auto deleteTmps = spnc::option::deleteTemporaryFiles.get(*config);
    // Collect graph statistics on transformed / canonicalized MLIR.
    auto statsFile = StatsFile(spnc::option::graphStatsFile.get(*config), deleteTmps);
    auto& graphStats = job->insertAction<CollectGraphStatistics>(lospnTransform, std::move(statsFile));
    // Join the two actions happening on the transformed module (Graph-Stats & SPN-to-Standard-MLIR lowering).
    auto& joinAction = job->insertAction<JoinAction<mlir::ModuleOp, StatsFile>>(lospnTransform, graphStats);
    lospnTransformResult = &joinAction;
  }
  auto& lospn2cpu = job->insertAction<LoSPNtoCPUConversion>(*lospnTransformResult, ctx, diagHandler);
  auto& cpu2llvm = job->insertAction<CPUtoLLVMConversion>(lospn2cpu, ctx, diagHandler);

  // Convert the MLIR module to a LLVM-IR module.
  auto& llvmConversion = job->insertAction<MLIRtoLLVMIRConversion>(cpu2llvm, ctx, targetMachine);

  // Translate the generated LLVM IR module to object code and write it to an object file.
  auto objectFile = FileSystem::createTempFile<FileType::OBJECT>(false);
  SPDLOG_INFO("Generating object file {}", objectFile.fileName());
  auto& emitObjectCode = job->insertAction<EmitObjectCode>(llvmConversion, std::move(objectFile), targetMachine);

  // Link generated object file into shared object.
  auto sharedObject = FileSystem::createTempFile<FileType::SHARED_OBJECT>(false);
  SPDLOG_INFO("Compiling to shared object file {}", sharedObject.fileName());
  // Add additional libraries to the link command if necessary.
  llvm::SmallVector<std::string, 3> additionalLibs;
  // Link vector libraries if specified by option.
  auto veclib = spnc::option::vectorLibrary.get(*config);
  if (veclib != spnc::option::VectorLibrary::NONE) {
    switch (veclib) {
      case spnc::option::VectorLibrary::SVML: additionalLibs.push_back("svml");
        break;
      case spnc::option::VectorLibrary::LIBMVEC: additionalLibs.push_back("m");
        break;
      default:SPNC_FATAL_ERROR("Unknown vector library");
    }
  }
  auto searchPaths = parseLibrarySearchPaths(spnc::option::searchPaths.get(*config));
  auto& linkSharedObject =
      job->insertFinalAction<ClangKernelLinking>(emitObjectCode, std::move(sharedObject), kernelInfo, 
                                                additionalLibs, searchPaths);
  return std::move(job);
}