//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/FileSystem.h>
#include <driver/BaseActions.h>
#include <driver/GlobalOptions.h>
#include <frontend/json/Parser.h>
#include <codegen/mlir/codegen/MLIRCodeGen.h>
#include <SPN/SPNDialect.h>
#include <codegen/mlir/pipeline/SPNDialectPipeline.h>
#include "codegen/mlir/conversion/SPNtoStandardConversion.h"
#include "codegen/mlir/conversion/SPNtoLLVMConversion.h"
#include "codegen/mlir/conversion/MLIRtoLLVMIRConversion.h"
#include "codegen/mlir/analysis/CollectGraphStatistics.h"
#include <driver/action/LLVMWriteBitcode.h>
#include <driver/action/LLVMStaticCompiler.h>
#include <driver/action/ClangKernelLinking.h>
#include "MLIRToolchain.h"
#include "mlir/InitAllDialects.h"

using namespace spnc;

std::unique_ptr<Job<Kernel> > MLIRToolchain::constructJobFromFile(const std::string& inputFile,
                                                                  const Configuration& config) {
  // Construct file input action.
  auto fileInput = std::make_unique<FileInputAction>(inputFile);
  return constructJob(std::move(fileInput), config);
}

std::unique_ptr<Job<Kernel> > MLIRToolchain::constructJobFromString(const std::string& inputString,
                                                                    const Configuration& config) {
  // Construct string input action.
  auto stringInput = std::make_unique<StringInputAction>(inputString);
  return constructJob(std::move(stringInput), config);
}

std::unique_ptr<Job<Kernel>> MLIRToolchain::constructJob(std::unique_ptr<ActionWithOutput<std::string>> input,
                                                         const Configuration& config) {
  std::unique_ptr<Job<Kernel>> job = std::make_unique<Job<Kernel>>();
  // Construct parser to parse JSON from input.
  auto graphIRContext = std::make_shared<GraphIRContext>();
  auto& parser = job->insertAction<Parser>(*input, graphIRContext);
  // Invoke MLIR code-generation on parsed tree.
  std::string kernelName = "spn_kernel";
  auto ctx = std::make_shared<MLIRContext>();
  mlir::registerAllDialects(ctx->getDialectRegistry());
  ctx->getDialectRegistry().insert<mlir::spn::SPNDialect>();
  ctx->loadDialect<mlir::spn::SPNDialect>();
  ctx->loadDialect<mlir::StandardOpsDialect>();
  ctx->loadDialect<mlir::LLVM::LLVMDialect>();
  auto& mlirCodeGen = job->insertAction<MLIRCodeGen>(parser, kernelName, ctx);
  auto& spnDialectPipeline = job->insertAction<SPNDialectPipeline>(mlirCodeGen, ctx);
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
      job->insertFinalAction<ClangKernelLinking>(compileObject, std::move(sharedObject), kernelName);
  job->addAction(std::move(input));
  return std::move(job);
}
