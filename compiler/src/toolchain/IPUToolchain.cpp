//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "IPUToolchain.h"
#include "Kernel.h"
#include "ipu/IPUTargetInfo.h"
#include "ipu/IPUTargetMachine.h"
#include "option/GlobalOptions.h"
#include "pipeline/BasicSteps.h"
#include "pipeline/Pipeline.h"
#include "pipeline/steps/codegen/EmitLLVMIR.h"
#include "pipeline/steps/codegen/EmitObjectCodeForIPU.h"
#include "pipeline/steps/frontend/SPFlowToMLIRDeserializer.h"
#include "pipeline/steps/linker/ClangKernelLinking.h"
#include "pipeline/steps/mlir/conversion/HiSPNtoLoSPNConversion.h"
#include "pipeline/steps/mlir/conversion/IPUtoLLVMConversion.h"
#include "pipeline/steps/mlir/conversion/LoSPNtoIPUConversion.h"
#include "pipeline/steps/mlir/conversion/MLIRtoLLVMIRConversion.h"
#include "pipeline/steps/mlir/transformation/LoSPNTransformations.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

using namespace spnc;
using namespace mlir;

std::unique_ptr<Pipeline<Kernel>> IPUToolchain::setupPipeline(const std::string &inputFile,
                                                              std::unique_ptr<interface::Configuration> config) {
  // Uncomment the following two lines to get detailed output during MLIR
  // dialect conversion;
  // llvm::DebugFlag = true;
  // llvm::setCurrentDebugType("dialect-conversion");

  SPDLOG_INFO("Setting up IPU pipeline for file {}", inputFile);

  // Initialize the pipeline.
  std::unique_ptr<Pipeline<Kernel>> pipeline = std::make_unique<Pipeline<Kernel>>();

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
  if (spnc::option::mcOptLevel.isPresent(*config) && spnc::option::mcOptLevel.get(*config) != mcOptLevel) {
    auto optionValue = spnc::option::mcOptLevel.get(*config);
    SPDLOG_INFO("Option mc-opt-level (value: {}) takes precedence over option "
                "opt-level (value: {})",
                optionValue, mcOptLevel);
    mcOptLevel = optionValue;
  }

  // Initialize kernel information.
  auto kernelInfo = std::make_unique<KernelInfo>();
  kernelInfo->target = KernelTarget::IPU;
  kernelInfo->ipuTarget = (IPUTarget)option::ipuTarget.get(*config);

  auto targetMachine = createTargetMachine(mcOptLevel, kernelInfo->ipuTarget);

  // Attach the LLVM target machine and the kernel information to the pipeline
  // context
  pipeline->getContext()->add(std::move(targetMachine));
  pipeline->getContext()->add(std::move(kernelInfo));

  // First step of the pipeline: Locate the input file.
  auto &locateInput = pipeline->emplaceStep<LocateFile<FileType::SPN_BINARY>>(inputFile);

  // Deserialize the SPFlow graph serialized via Cap'n Proto to MLIR.
  auto &deserialized = pipeline->emplaceStep<SPFlowToMLIRDeserializer>(locateInput);

  // Convert from HiSPN dialect to LoSPN.
  auto &hispn2lospn = pipeline->emplaceStep<HiSPNtoLoSPNConversion>(deserialized);
  // Perform transformations on the LoSPN dialect module.
  auto &lospnTransform = pipeline->emplaceStep<LoSPNTransformations>(hispn2lospn);
  // Lower from LoSPN to upstream dialects to target CPU, including
  // vectorization.
  auto &lospn2ipu = pipeline->emplaceStep<LoSPNtoIPUConversion>(lospnTransform);
  // Convert from mixture of upstream dialects to LLVM dialect.
  auto &ipu2llvm = pipeline->emplaceStep<IPUtoLLVMConversion>(lospn2ipu);

  // Convert the MLIR module to a LLVM-IR module.
  auto &llvmConversion = pipeline->emplaceStep<MLIRtoLLVMIRConversion>(ipu2llvm);
  // Store the generated LLVM IR module in a temporary file.
  auto &llvmIR = pipeline->emplaceStep<CreateTmpFile<FileType::LLVM_IR>>(true);
  pipeline->emplaceStep<EmitLLVMIR>(llvmConversion, llvmIR);
  // Compile the generated LLVM IR file to object code and write it to another file.
  auto &graphPropgram = pipeline->emplaceStep<CreateTmpFile<FileType::GRAPH_PROGRAM>>(true);
  pipeline->emplaceStep<EmitObjectCodeForIPU<FileType::LLVM_IR>>(llvmIR, graphPropgram);

  // // Link generated object file into shared object.
  // auto& sharedObject = pipeline->emplaceStep < CreateTmpFile <
  // FileType::SHARED_OBJECT >> (false);
  // // Add additional libraries to the link command if necessary.
  // llvm::SmallVector<std::string, 3> additionalLibs;
  // // Link vector libraries if specified by option.
  // auto veclib = spnc::option::vectorLibrary.get(*config);
  // if (!validateVectorLibrary(*config)) {
  //   SPDLOG_WARN("Vector library selection is invalid on this platform,
  //   overriding with 'None'"); veclib = spnc::option::VectorLibrary::NONE;
  // }
  // if (veclib != spnc::option::VectorLibrary::NONE) {
  //   switch (veclib) {
  //     case spnc::option::VectorLibrary::SVML:
  //     additionalLibs.push_back("svml");
  //       break;
  //     case spnc::option::VectorLibrary::LIBMVEC:
  //     additionalLibs.push_back("m");
  //       break;
  //     case spnc::option::VectorLibrary::ARM:
  //     additionalLibs.push_back("mathlib");
  //       break;
  //     default:SPNC_FATAL_ERROR("Unknown vector library");
  //   }
  // }
  // auto searchPaths =
  // parseLibrarySearchPaths(spnc::option::searchPaths.get(*config)); auto
  // libraryInfo = std::make_unique<LibraryInfo>(additionalLibs, searchPaths);
  // pipeline->getContext()->add(std::move(libraryInfo));
  // // Link the kernel with the libraries to produce executable (shared
  // object). (void) pipeline->emplaceStep<ClangKernelLinking>(emitObjectCode,
  // sharedObject);
  // // Add the CLI configuration to the pipeline context.
  pipeline->getContext()->add(std::move(config));

  return pipeline;
}

// bool CPUToolchain::validateVectorLibrary(interface::Configuration& config) {
//   auto veclib = spnc::option::vectorLibrary.get(config);
//   auto& targetInfo = mlir::spn::TargetInformation::nativeCPUTarget();
//   switch (veclib) {
//     case spnc::option::VectorLibrary::SVML:
//     case spnc::option::VectorLibrary::LIBMVEC:return
//     targetInfo.isX8664Target(); case spnc::option::VectorLibrary::ARM:return
//     targetInfo.isAARCH64Target(); default:return false;
//   }
// }

std::unique_ptr<llvm::TargetMachine> spnc::IPUToolchain::createTargetMachine(int optLevel, IPUTarget ipuTarget) {
  llvm::Triple targetTriple;
  llvm::StringRef cpu;
  llvm::SubtargetFeatures features;
  if (ipuTarget == IPUTarget::Model) {
    // Initialize the native target machine
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();

    cpu = llvm::sys::getHostCPUName();
    targetTriple = llvm::Triple{llvm::sys::getDefaultTargetTriple()};
    llvm::StringMap<bool> hostFeatures;
    if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
      for (auto &f : hostFeatures) {
        features.AddFeature(f.first(), f.second);
      }
    }

  } else {
    // Initialize the IPU target machine
    initializeIPUTargetInfo();
    initializeIPUTarget();
    targetTriple = llvm::Triple{"colossus-graphcore-unknown-elf"};

    switch (ipuTarget) {
    case IPU1:
      cpu = "ipu1";
      break;
    case IPU2:
      cpu = "ipu2";
      break;
    case IPU21:
      cpu = "ipu21";
      break;
    case Model:
      // Handled above
      break;
    }
    features.AddFeature(cpu, true);
    features.AddFeature("worker", true);
  }

  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple.getTriple(), errorMessage);
  if (!target) {
    SPNC_FATAL_ERROR("No target for target triple {}: {}", targetTriple.getTriple(), errorMessage);
  }

  SPDLOG_INFO("Target machine triple: {}", targetTriple.getTriple());
  SPDLOG_INFO("Target machine CPU name: {}", cpu.str());
  SPDLOG_INFO("Target machine features: {}", features.getString());

  llvm::CodeGenOpt::Level cgOptLevel = llvm::CodeGenOpt::Default;
  switch (optLevel) {
  case 0:
    cgOptLevel = llvm::CodeGenOpt::None;
    break;
  case 1:
    cgOptLevel = llvm::CodeGenOpt::Less;
    break;
  case 2:
    cgOptLevel = llvm::CodeGenOpt::Default;
    break;
  case 3:
    cgOptLevel = llvm::CodeGenOpt::Aggressive;
    break;
  default:
    SPNC_FATAL_ERROR("Invalid optimization level {}", optLevel);
  }

  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      targetTriple.getTriple(), cpu, features.getString(), {}, llvm::Reloc::PIC_, llvm::None, cgOptLevel)};
  return machine;
}