//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/FileSystem.h>
#include <driver/BaseActions.h>
#include <frontend/json/Parser.h>
#include <graph-ir/transform/BinaryTreeTransform.h>
#include <codegen/mlir/MLIRCodeGen.h>
#include <codegen/mlir/pipeline/MLIRPipeline.h>
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

  // Run the MLIR-based pipeline, including progressive lowering to LLVM dialect.
  auto& mlirPipeline = job->insertAction<MLIRPipeline>(mlirCodeGen, ctx);
  // Convert the MLIR module to a LLVM IR module.
  auto& llvmConversion = job->insertAction<MLIRtoLLVMConversion>(mlirPipeline, ctx);
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
