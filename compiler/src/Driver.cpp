//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <spnc.h>
#include "toolchain/CPUToolchain.h"
#include <option/Options.h>
#include <option/GlobalOptions.h>
#include <util/Logging.h>
#include <TargetInformation.h>
#if SPNC_CUDA_SUPPORT
// Only include if CUDA GPU support was enabled.
#include "toolchain/CUDAGPUToolchain.h"
#endif
#include "toolchain/FPGAToolchain.h"

using namespace spnc;

Kernel spn_compiler::compileQuery(const std::string& inputFile, const options_t& options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");
  for (const auto& [k, v] : options)
    std::cout << k << ": " << v << "\n";
  auto config = interface::Options::parse(options);
  std::unique_ptr<Pipeline<Kernel>> pipeline;
  if (spnc::option::compilationTarget.get(*config) == option::TargetMachine::CUDA) {
#if SPNC_CUDA_SUPPORT
    pipeline = CUDAGPUToolchain::setupPipeline(inputFile, std::move(config));
#else
    SPNC_FATAL_ERROR("Target was 'CUDA', but the compiler does not support CUDA GPUs. "
                     "Enable with CUDA_GPU_SUPPORT=ON during build")
#endif
  } else if (spnc::option::compilationTarget.get(*config) == option::TargetMachine::FPGA) {
    pipeline = FPGAToolchain::setupPipeline(inputFile, std::move(config));
  } else {
    pipeline = CPUToolchain::setupPipeline(inputFile, std::move(config));
  }
  SPDLOG_INFO("Executing compilation pipeline: {}", pipeline->toText());
  auto result = pipeline->execute();
  if (failed(result)) {
    SPNC_FATAL_ERROR("Execution of the compilation pipeline stopped with message: {}", result.message());
  }
  auto kernel = pipeline->result();

  if (kernel->getKernelType() == KernelType::CLASSICAL_KERNEL) {
    ClassicalKernel classical = kernel->getClassicalKernel();
    SPDLOG_INFO("Generated Kernel in {}, kernel name {}", classical.fileName, classical.kernelName);
  } else if (kernel->getKernelType() == KernelType::FPGA_KERNEL) {
    FPGAKernel fpga = kernel->getFPGAKernel();
    SPDLOG_INFO("Generated Kernel in {}, kernel name {}", fpga.fileName, fpga.kernelName);
  } else {
    assert(false);
  }

  return *kernel;
}


bool spn_compiler::isTargetSupported(const std::string& target){
  if(target == "CPU"){
    return true;
  }
  if(target == "CUDA"){
    #if SPNC_CUDA_SUPPORT
    return true;
    #else
    return false;
    #endif
  }
  if (target == "FPGA") {
    return true;
  }
  return false;
}

bool spn_compiler::isFeatureSupported(const std::string& feature){
  if(feature == "vectorize"){
      auto& targetInfo = mlir::spn::TargetInformation::nativeCPUTarget();
      return targetInfo.hasAVXSupport() ||
          targetInfo.hasAVX2Support() || targetInfo.hasAVX512Support() ||
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