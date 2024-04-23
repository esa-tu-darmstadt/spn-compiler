//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "CPUToolchain.h"
#include "TargetInformation.h"
#include "pipeline/BasicSteps.h"
#include "pipeline/Pipeline.h"
#include "pipeline/steps/codegen/EmitObjectCode.h"
#include "pipeline/steps/frontend/SPFlowToMLIRDeserializer.h"
#include "pipeline/steps/linker/ClangKernelLinking.h"
#include "pipeline/steps/mlir/conversion/CPUtoLLVMConversion.h"
#include "pipeline/steps/mlir/conversion/HiSPNtoLoSPNConversion.h"
#include "pipeline/steps/mlir/conversion/LoSPNtoCPUConversion.h"
#include "pipeline/steps/mlir/conversion/MLIRtoLLVMIRConversion.h"
#include "pipeline/steps/mlir/transformation/LoSPNTransformations.h"

using namespace spnc;
using namespace mlir;

std::unique_ptr<Pipeline<Kernel>>
CPUToolchain::setupPipeline(const std::string &inputFile,
                            std::unique_ptr<interface::Configuration> config) {
  // Uncomment the following two lines to get detailed output during MLIR
  // dialect conversion;
  // llvm::DebugFlag = true;
  // llvm::setCurrentDebugType("dialect-conversion");

  // Initialize the pipeline.
  std::unique_ptr<Pipeline<Kernel>> pipeline =
      std::make_unique<Pipeline<Kernel>>();

  // Initialize the MLIR context.
  auto ctx = std::make_unique<MLIRContext>();
  initializeMLIRContext(*ctx);
  // If IR should be dumped between steps/passes, we need to disable
  // multi-threading in MLIR
  if (spnc::option::dumpIR.get(*config)) {
    ctx->enableMultithreading(false);
  }
  auto diagHandler = setupDiagnosticHandler(ctx.get());
  // Attach MLIR context and diagnostics handler to pipeline context
  pipeline->getContext()->add(std::move(diagHandler));
  pipeline->getContext()->add(std::move(ctx));

  // Create an LLVM target machine and set the optimization level.
  int mcOptLevel = spnc::option::optLevel.get(*config);
  if (spnc::option::mcOptLevel.isPresent(*config) &&
      spnc::option::mcOptLevel.get(*config) != mcOptLevel) {
    auto optionValue = spnc::option::mcOptLevel.get(*config);
    SPDLOG_INFO("Option mc-opt-level (value: {}) takes precedence over option "
                "opt-level (value: {})",
                optionValue, mcOptLevel);
    mcOptLevel = optionValue;
  }
  auto targetMachine = createTargetMachine(mcOptLevel);
  // Initialize kernel information.
  auto kernelInfo = std::make_unique<KernelInfo>();
  kernelInfo->target = KernelTarget::CPU;
  // Attach the LLVM target machine and the kernel information to the pipeline
  // context
  pipeline->getContext()->add(std::move(targetMachine));
  pipeline->getContext()->add(std::move(kernelInfo));

  // First step of the pipeline: Locate the input file.
  auto &locateInput =
      pipeline->emplaceStep<LocateFile<FileType::SPN_BINARY>>(inputFile);

  // Deserialize the SPFlow graph serialized via Cap'n Proto to MLIR.
  auto &deserialized =
      pipeline->emplaceStep<SPFlowToMLIRDeserializer>(locateInput);

  // Convert from HiSPN dialect to LoSPN.
  auto &hispn2lospn =
      pipeline->emplaceStep<HiSPNtoLoSPNConversion>(deserialized);
  // Perform transformations on the LoSPN dialect module.
  auto &lospnTransform =
      pipeline->emplaceStep<LoSPNTransformations>(hispn2lospn);
  // Lower from LoSPN to upstream dialects to target CPU, including
  // vectorization.
  auto &lospn2cpu = pipeline->emplaceStep<LoSPNtoCPUConversion>(lospnTransform);
  // Convert from mixture of upstream dialects to LLVM dialect.
  auto &cpu2llvm = pipeline->emplaceStep<CPUtoLLVMConversion>(lospn2cpu);

  // Convert the MLIR module to a LLVM-IR module.
  auto &llvmConversion =
      pipeline->emplaceStep<MLIRtoLLVMIRConversion>(cpu2llvm);

  // Translate the generated LLVM IR module to object code and write it to an
  // object file.
  auto &objectFile =
      pipeline->emplaceStep<CreateTmpFile<FileType::OBJECT>>(true);
  auto &emitObjectCode =
      pipeline->emplaceStep<EmitObjectCode>(llvmConversion, objectFile);

  // Link generated object file into shared object.
  auto &sharedObject =
      pipeline->emplaceStep<CreateTmpFile<FileType::SHARED_OBJECT>>(false);
  // Add additional libraries to the link command if necessary.
  llvm::SmallVector<std::string, 3> additionalLibs;
  // Link vector libraries if specified by option.
  auto veclib = spnc::option::vectorLibrary.get(*config);
  if (!validateVectorLibrary(*config)) {
    SPDLOG_WARN("Vector library selection is invalid on this platform, "
                "overriding with 'None'");
    veclib = spnc::option::VectorLibrary::NONE;
  }
  if (veclib != spnc::option::VectorLibrary::NONE) {
    switch (veclib) {
    case spnc::option::VectorLibrary::SVML:
      additionalLibs.push_back("svml");
      break;
    case spnc::option::VectorLibrary::LIBMVEC:
      additionalLibs.push_back("m");
      break;
    case spnc::option::VectorLibrary::ARM:
      additionalLibs.push_back("mathlib");
      break;
    default:
      SPNC_FATAL_ERROR("Unknown vector library");
    }
  }
  auto searchPaths =
      parseLibrarySearchPaths(spnc::option::searchPaths.get(*config));
  auto libraryInfo = std::make_unique<LibraryInfo>(additionalLibs, searchPaths);
  pipeline->getContext()->add(std::move(libraryInfo));
  // Link the kernel with the libraries to produce executable (shared object).
  (void)pipeline->emplaceStep<ClangKernelLinking>(emitObjectCode, sharedObject);
  // Add the CLI configuration to the pipeline context.
  pipeline->getContext()->add(std::move(config));

  return pipeline;
}

bool CPUToolchain::validateVectorLibrary(interface::Configuration &config) {
  auto veclib = spnc::option::vectorLibrary.get(config);
  auto &targetInfo = mlir::spn::TargetInformation::nativeCPUTarget();
  switch (veclib) {
  case spnc::option::VectorLibrary::SVML:
  case spnc::option::VectorLibrary::LIBMVEC:
    return targetInfo.isX8664Target();
  case spnc::option::VectorLibrary::ARM:
    return targetInfo.isAARCH64Target();
  default:
    return false;
  }
}