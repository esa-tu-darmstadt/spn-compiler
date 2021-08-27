//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "CUDAGPUToolchain.h"
#include <driver/BaseActions.h>
#include "codegen/mlir/conversion/HiSPNtoLoSPNConversion.h"
#include "codegen/mlir/conversion/LoSPNtoGPUConversion.h"
#include "codegen/mlir/conversion/GPUtoLLVMConversion.h"
#include "codegen/mlir/conversion/MLIRtoLLVMIRConversion.h"
#include <driver/action/ClangKernelLinking.h>
#include <codegen/mlir/frontend/MLIRDeserializer.h>
#include <codegen/mlir/transformation/LoSPNTransformations.h>
#include <driver/action/EmitObjectCode.h>

#ifndef SPNC_CUDA_RUNTIME_WRAPPERS_DIR
// This define should usually be set by CMake, pointing
// to the correct location of the MLIR CUDA runtime wrappers.
#define SPNC_CUDA_RUNTIME_WRAPPERS_DIR "/usr/local"
#endif

using namespace spnc;
using namespace mlir;

std::unique_ptr<Job<Kernel> > CUDAGPUToolchain::constructJobFromFile(const std::string& inputFile,
                                                const std::shared_ptr<interface::Configuration>& config) {
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
  int mcOptLevel = spnc::option::optLevel.get(*config);
  if (spnc::option::mcOptLevel.isPresent(*config) && spnc::option::mcOptLevel.get(*config) != mcOptLevel) {
    auto optionValue = spnc::option::mcOptLevel.get(*config);
    SPDLOG_INFO("Option mc-opt-level (value: {}) takes precedence over option opt-level (value: {})",
                optionValue, mcOptLevel);
    mcOptLevel = optionValue;
  }
  auto targetMachine = createTargetMachine(mcOptLevel);
  auto kernelInfo = std::make_shared<KernelInfo>();
  kernelInfo->target = KernelTarget::CUDA;
  BinarySPN binarySPNFile{inputFile, false};
  auto& deserialized = job->insertAction<MLIRDeserializer>(std::move(binarySPNFile), ctx, kernelInfo);
  auto& hispn2lospn = job->insertAction<HiSPNtoLoSPNConversion>(deserialized, ctx, diagHandler);
  auto& lospnTransform = job->insertAction<LoSPNTransformations>(hispn2lospn, ctx, diagHandler, kernelInfo);
  auto& lospn2gpu = job->insertAction<LoSPNtoGPUConversion>(lospnTransform, ctx, diagHandler);

  int irOptLevel = spnc::option::optLevel.get(*config);
  if (spnc::option::irOptLevel.isPresent(*config) && spnc::option::irOptLevel.get(*config) != irOptLevel) {
    auto optionValue = spnc::option::irOptLevel.get(*config);
    SPDLOG_INFO("Option ir-opt-level (value: {}) takes precedence over option opt-level (value: {})",
                optionValue, irOptLevel);
    irOptLevel = optionValue;
  }

  // Convert the GPU portion of the code to CUBIN.
  auto& cpu2llvm = job->insertAction<GPUtoLLVMConversion>(lospn2gpu, ctx, irOptLevel);

  // Convert the remaining MLIR module to a LLVM-IR module.
  auto& llvmConversion = job->insertAction<MLIRtoLLVMIRConversion>(cpu2llvm, ctx, targetMachine, irOptLevel);

  // Translate the generated LLVM IR module to object code and write it to an object file.
  auto objectFile = FileSystem::createTempFile<FileType::OBJECT>(false);
  SPDLOG_INFO("Generating object file {}", objectFile.fileName());
  auto& emitObjectCode = job->insertAction<EmitObjectCode>(llvmConversion, std::move(objectFile), targetMachine);

  // Link generated object file into shared object.
  auto sharedObject = FileSystem::createTempFile<FileType::SHARED_OBJECT>(false);
  SPDLOG_INFO("Compiling to shared object file {}", sharedObject.fileName());
  // The generated kernel must be linked against the MLIR CUDA runtime wrappers.
  llvm::SmallVector<std::string, 3> additionalLibs;
  additionalLibs.push_back("spnc-cuda-wrappers");
  auto searchPaths = parseLibrarySearchPaths(spnc::option::searchPaths.get(*config));
  searchPaths.push_back(SPNC_CUDA_RUNTIME_WRAPPERS_DIR);
  (void) job->insertFinalAction<ClangKernelLinking>(emitObjectCode, std::move(sharedObject), kernelInfo,
                                                 additionalLibs, searchPaths);
  return job;
}