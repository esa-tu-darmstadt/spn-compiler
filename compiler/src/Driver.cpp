//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/LoSPNPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUPipeline.h"
#include "toolchain/CPUToolchain.h"
#include "toolchain/IPUToolchain.h"
#include <TargetInformation.h>
#include <option/Options.h>
#include <spnc.h>
#include <util/Logging.h>
#if SPNC_CUDA_SUPPORT
// Only include if CUDA GPU support was enabled.
#include "toolchain/CUDAGPUToolchain.h"
#endif

using namespace spnc;

namespace {
/// Processes the options, setting all registered CL options.
void parseOptions(const options_t &options) {
  std::vector<std::string> args;

  // The first argument is the program name.
  args.push_back("spnc");
  for (const auto &option : options) {
    if (option.second.empty())
      args.push_back("--" + option.first);
    else
      args.push_back("--" + option.first + "=" + option.second);

    SPDLOG_INFO("Option: {}={}", option.first, option.second);
  }

  // ParseCommandLineOptions expects a vector of C-strings.
  std::vector<const char *> argv;
  for (const auto &arg : args) {
    argv.push_back(arg.c_str());
  }

  llvm::cl::ParseCommandLineOptions((int)argv.size(), argv.data(), "SPN Compiler");
}
} // namespace

Kernel spn_compiler::compileQuery(const std::string &inputFile, const options_t &options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");

  mlir::spn::low::registerLoSPNPasses();

  // Parse the options if there are any.
  if (!options.empty())
    parseOptions(options);

  std::unique_ptr<Pipeline<Kernel>> pipeline;
  auto target = spnc::option::compilationTarget.get(*config);
  if (target == option::TargetMachine::CUDA) {
#if SPNC_CUDA_SUPPORT
    pipeline = CUDAGPUToolchain::setupPipeline(inputFile);
#else
    SPNC_FATAL_ERROR("Target was 'CUDA', but the compiler does not support CUDA GPUs. "
                     "Enable with CUDA_GPU_SUPPORT=ON during build")
#endif
  } else if (target == option::TargetMachine::IPU) {
#if SPNC_IPU_SUPPORT
    pipeline = IPUToolchain::setupPipeline(inputFile, std::move(config));
#else
    SPNC_FATAL_ERROR("Target was 'IPU', but the compiler does not support IPUs. "
                     "Enable with IPU_SUPPORT=ON during build")
#endif
  } else {
    pipeline = CPUToolchain::setupPipeline(inputFile);
  }
  SPDLOG_INFO("Executing compilation pipeline: {}", pipeline->toText());
  auto result = pipeline->execute();
  if (failed(result)) {
    SPNC_FATAL_ERROR("Execution of the compilation pipeline stopped with message: {}", result.message());
  }
  auto kernel = pipeline->result();
  SPDLOG_INFO("Generated Kernel in {}, kernel name {}", kernel->fileName(), kernel->kernelName());
  return *kernel;
}

bool spn_compiler::isTargetSupported(const std::string &target) {
  if (target == "CPU") {
    return true;
  }
  if (target == "CUDA") {
#if SPNC_CUDA_SUPPORT
    return true;
#else
    return false;
#endif
  }
  if (target == "IPU") {
#if SPNC_IPU_SUPPORT
    return true;
#else
    return false;
#endif
  }
  return false;
}

bool spn_compiler::isFeatureSupported(const std::string &feature) {
  if (feature == "vectorize") {
    auto &targetInfo = mlir::spn::TargetInformation::nativeCPUTarget();
    return targetInfo.hasAVXSupport() || targetInfo.hasAVX2Support() || targetInfo.hasAVX512Support() ||
           targetInfo.hasNeonSupport();
  }
  if (feature == "AVX") {
    return mlir::spn::TargetInformation::nativeCPUTarget().hasAVXSupport();
  }
  if (feature == "AVX2") {
    return mlir::spn::TargetInformation::nativeCPUTarget().hasAVX2Support();
  }
  if (feature == "AVX512") {
    return mlir::spn::TargetInformation::nativeCPUTarget().hasAVX512Support();
  }
  if (feature == "Neon") {
    return mlir::spn::TargetInformation::nativeCPUTarget().hasNeonSupport();
  }
  // TODO Add query support for more features.
  return false;
}

std::string spn_compiler::getHostArchitecture() {
  return mlir::spn::TargetInformation::nativeCPUTarget().getHostArchitecture();
}