//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/FileSystem.h>
#include <driver/BaseActions.h>
#include <driver/GlobalOptions.h>
#include <frontend/json/Parser.h>
#include <graph-ir/transform/BinaryTreeTransform.h>
#include <codegen/mlir/MLIRCodeGen.h>
#include <codegen/mlir/lowering/action/SPNToLLVMLowering.h>
#include <codegen/mlir/lowering/action/SPNToStandardLowering.h>
#include <codegen/mlir/pipeline/MLIRPipeline.h>
#include <codegen/mlir/util/action/GraphStatsCollection.h>
#include <driver/action/MLIRtoLLVMConversion.h>
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
  std::unique_ptr<Job<Kernel>> job{new Job<Kernel>()};
  // Construct parser to parse JSON from input.
  auto graphIRContext = std::make_shared<GraphIRContext>();
  auto& parser = job->insertAction<Parser>(*input, graphIRContext);
  // Invoke MLIR code-generation on parsed tree.
  std::string kernelName = "spn_kernel";
  mlir::registerAllDialects();
  // Register our Dialect with MLIR.
  mlir::registerDialect<mlir::spn::SPNDialect>();
  auto ctx = std::make_shared<MLIRContext>();
  auto& mlirCodeGen = job->insertAction<MLIRCodeGen>(parser, kernelName, ctx);
  // Run the MLIR-based pipeline, i.e. simplification and canonicalization.
  auto& mlirPipeline = job->insertAction<MLIRPipeline>(mlirCodeGen, ctx);
  // Lower the SPN-MLIR dialect to Standard-MLIR.
  auto& standardDialect = job->insertAction<SPNToStandardLowering>(mlirPipeline, ctx);
  ActionWithOutput<mlir::ModuleOp>* standardDialectResult = &standardDialect;

  // If requested via the configuration, collect graph statistics.
  if (spnc::option::collectGraphStats.get(config)) {
    auto deleteTmps = spnc::option::deleteTemporaryFiles.get(config);
    // Collect graph statistics on transformed / canonicalized MLIR.
    auto statsFile = StatsFile(spnc::option::graphStatsFile.get(config), deleteTmps);
    auto& graphStats = job->insertAction<GraphStatsCollection>(mlirPipeline, std::move(statsFile));
    // Join the two actions happening on the transformed module (Graph-Stats & SPN-to-Standard-MLIR lowering).
    auto& joinAction = job->insertAction<JoinAction<mlir::ModuleOp, StatsFile>>(standardDialect, graphStats);
    standardDialectResult = &joinAction;
  }

  // Lower the Standard-MLIR dialect to LLVM-MLIR.
  auto& llvmDialect = job->insertAction<SPNToLLVMLowering>(*standardDialectResult, ctx);
  // Convert the MLIR module to a LLVM-IR module.
  auto& llvmConversion = job->insertAction<MLIRtoLLVMConversion>(llvmDialect, ctx);

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
