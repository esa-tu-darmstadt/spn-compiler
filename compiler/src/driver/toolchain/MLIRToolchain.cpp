//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/FileSystem.h>
#include <driver/BaseActions.h>
#include <driver/GlobalOptions.h>
#include <SPN/SPNDialect.h>
#include <codegen/mlir/pipeline/SPNDialectPipeline.h>
#include "codegen/mlir/conversion/SPNtoStandardConversion.h"
#include "codegen/mlir/conversion/SPNtoLLVMConversion.h"
#include "codegen/mlir/conversion/MLIRtoLLVMIRConversion.h"
#include "codegen/mlir/analysis/CollectGraphStatistics.h"
#include <driver/action/LLVMWriteBitcode.h>
#include <driver/action/LLVMStaticCompiler.h>
#include <driver/action/ClangKernelLinking.h>
#include <codegen/mlir/frontend/MLIRDeserializer.h>
#include "MLIRToolchain.h"
#include "mlir/InitAllDialects.h"

using namespace spnc;

std::unique_ptr<Job<Kernel> > MLIRToolchain::constructJobFromFile(const std::string& inputFile,
                                                                  const Configuration& config) {
  std::unique_ptr<Job<Kernel>> job = std::make_unique<Job<Kernel>>();
  // Invoke MLIR code-generation on parsed tree.
  auto ctx = std::make_shared<MLIRContext>();
  auto kernelInfo = std::make_shared<KernelInfo>();
  mlir::registerAllDialects(ctx->getDialectRegistry());
  ctx->getDialectRegistry().insert<mlir::spn::SPNDialect>();
  ctx->loadDialect<mlir::spn::SPNDialect>();
  ctx->loadDialect<mlir::StandardOpsDialect>();
  ctx->loadDialect<mlir::LLVM::LLVMDialect>();
  BinarySPN binarySPNFile{inputFile, false};
  auto& deserialized = job->insertAction<MLIRDeserializer>(std::move(binarySPNFile), ctx, kernelInfo);
  auto& spnDialectPipeline = job->insertAction<SPNDialectPipeline>(deserialized, ctx);
  ActionWithOutput<ModuleOp>* spnPipelineResult = &spnDialectPipeline;
  // If requested via the configuration, collect graph statistics.
  if (spnc::option::collectGraphStats.get(config)) {
    auto deleteTmps = spnc::option::deleteTemporaryFiles.get(config);
    // Collect graph statistics on transformed / canonicalized MLIR.
    auto statsFile = StatsFile(spnc::option::graphStatsFile.get(config), deleteTmps);
    auto& graphStats = job->insertAction<CollectGraphStatistics>(spnDialectPipeline, std::move(statsFile));
    // Join the two actions happening on the transformed module (Graph-Stats & SPN-to-Standard-MLIR lowering).
    auto& joinAction = job->insertAction<JoinAction<mlir::ModuleOp, StatsFile>>(spnDialectPipeline, graphStats);
    spnPipelineResult = &joinAction;
  }

  auto& spn2standard = job->insertAction<SPNtoStandardConversion>(*spnPipelineResult, ctx);
  auto& spn2llvm = job->insertAction<SPNtoLLVMConversion>(spn2standard, ctx);

  // Convert the MLIR module to a LLVM-IR module.
  auto& llvmConversion = job->insertAction<MLIRtoLLVMIRConversion>(spn2llvm, ctx);

  // Write generated LLVM module to bitcode-file.
  auto bitCodeFile = FileSystem::createTempFile<FileType::LLVM_BC>();
  auto& writeBitcode = job->insertAction<LLVMWriteBitcode>(llvmConversion, std::move(bitCodeFile));
  // Compile generated bitcode-file to object file.
  auto objectFile = FileSystem::createTempFile<FileType::OBJECT>();
  auto& compileObject = job->insertAction<LLVMStaticCompiler>(writeBitcode, std::move(objectFile));
  // Link generated object file into shared object.
  auto sharedObject = FileSystem::createTempFile<FileType::SHARED_OBJECT>(false);
  SPDLOG_INFO("Compiling to shared object file {}", sharedObject.fileName());
  auto& linkSharedObject =
      job->insertFinalAction<ClangKernelLinking>(compileObject, std::move(sharedObject), kernelInfo);
  //job->addAction(std::move(input));
  return std::move(job);
}
