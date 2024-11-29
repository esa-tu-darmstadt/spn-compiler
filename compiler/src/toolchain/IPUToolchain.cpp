//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "IPUToolchain.h"
#include "Kernel.h"
#include "option/Options.h"
#include "pipeline/BasicSteps.h"
#include "pipeline/Pipeline.h"
#include "pipeline/steps/frontend/SPFlowToMLIRDeserializer.h"
#include "pipeline/steps/mlir/conversion/HiSPNtoLoSPNConversion.h"
#include "pipeline/steps/mlir/conversion/Poplar/CompilePoplarDialect.h"
#include "pipeline/steps/mlir/conversion/Poplar/LoSPNtoPoplarDialectConversion.h"
#include "pipeline/steps/mlir/transformation/LoSPNTransformations.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <memory>
#include <poplar/IPUModel.hpp>

using namespace spnc;
using namespace mlir;

std::unique_ptr<Pipeline<std::unique_ptr<Kernel>>>
IPUToolchain::setupPipeline(const std::string &inputFile) {
  // Uncomment the following two lines to get detailed output during MLIR
  // dialect conversion;
  // llvm::DebugFlag = true;
  // llvm::setCurrentDebugType("dialect-conversion");

  SPDLOG_INFO("Setting up IPU pipeline for file {}", inputFile);

  // Initialize the pipeline.
  auto pipeline = std::make_unique<Pipeline<std::unique_ptr<Kernel>>>();

  // Initialize the MLIR context.
  auto ctx = std::make_unique<MLIRContext>();
  initializeMLIRContext(*ctx);
  // If IR should be dumped between steps/passes, we need to disable
  // multi-threading in MLIR
  if (spnc::option::dumpIR) {
    ctx->enableMultithreading(false);
  }
  auto diagHandler = setupDiagnosticHandler(ctx.get());
  // std::unique_ptr<TargetExecutionModel> targetModel =
  // std::make_unique<IPUTargetExecutionModel>(); Attach MLIR context and
  // diagnostics handler to pipeline context
  pipeline->getContext()->add(std::move(diagHandler));
  pipeline->getContext()->add(std::move(ctx));
  // pipeline->getContext()->add(std::move(targetModel));

  // Create an LLVM target machine and set the optimization level.
  int mcOptLevel = spnc::option::optLevel;
  if (spnc::option::mcOptLevel.getNumOccurrences() > 0 &&
      spnc::option::mcOptLevel != mcOptLevel) {
    int optionValue = spnc::option::mcOptLevel;
    SPDLOG_INFO("Option mc-opt-level (value: {}) takes precedence over option "
                "opt-level (value: {})",
                optionValue, mcOptLevel);
    mcOptLevel = optionValue;
  }

  // Initialize kernel information.
  auto kernelInfo = std::make_unique<KernelInfo>();
  kernelInfo->target = KernelTarget::IPU;
  kernelInfo->ipuTarget = option::ipuTarget;

  // Attach the LLVM target machine and the kernel information to the pipeline
  // context
  pipeline->getContext()->add(std::move(kernelInfo));

  // Create the Poplar device and initialize the LLVM target.
  {
    ::poplar::Device device;

    if (option::ipuTarget == IPUTarget::Model) {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmParser();
      llvm::InitializeNativeTargetAsmPrinter();

      ::poplar::IPUModel ipuModel;
      device = ipuModel.createDevice();
    } else {
      LLVMInitializeColossusTargetInfo();
      LLVMInitializeColossusTarget();
      LLVMInitializeColossusTargetMC();

      llvm_unreachable("nyi");
    }
    ::poplar::Target target = device.getTarget();
    pipeline->getContext()->add<::poplar::Device>(std::move(device));
    pipeline->getContext()->add<::poplar::Target>(std::move(target));
  }

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
  // Lower from LoSPN to upstream dialects to the Poplar dialect
  auto &lospn2poplarMLIR =
      pipeline->emplaceStep<LoSPNtoPoplarDialectConversion>(lospnTransform);
  // Compile the Poplar dialect to a Poplar executable.
  auto &compilePoplarDialect =
      pipeline->emplaceStep<CompilePoplarDialect>(lospn2poplarMLIR);

  return pipeline;
}