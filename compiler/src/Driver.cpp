//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <spnc.h>
#include <driver/toolchain/CPUToolchain.h>
#include <driver/Options.h>
#include <driver/GlobalOptions.h>
#include <util/Logging.h>
#include <TargetInformation.h>
#if SPNC_CUDA_SUPPORT
// Only include if CUDA GPU support was enabled.
#include <driver/toolchain/CUDAGPUToolchain.h>
#endif

using namespace spnc;

Kernel spn_compiler::compileQuery(const std::string& inputFile, const options_t& options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");
  auto config = interface::Options::parse(options);
  std::unique_ptr<Job<Kernel>> job;
  if (spnc::option::compilationTarget.get(*config) == option::TargetMachine::CUDA) {
#if SPNC_CUDA_SUPPORT
    job = CUDAGPUToolchain::constructJobFromFile(inputFile, config);
#else
    SPNC_FATAL_ERROR("Target was 'CUDA', but the compiler does not support CUDA GPUs. "
                     "Enable with CUDA_GPU_SUPPORT=ON during build")
#endif
  } else {
    job = CPUToolchain::constructJobFromFile(inputFile, config);
  }
  auto kernel = job->execute();
  SPDLOG_INFO("Generated Kernel in {}, kernel name {}", kernel.fileName(), kernel.kernelName());
  return kernel;
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
  return false;
}

bool spn_compiler::isFeatureSupported(const std::string& feature){
  if(feature == "vectorize"){
      auto& targetInfo = mlir::spn::TargetInformation::nativeCPUTarget();
      return targetInfo.hasAVXSupport() || 
              targetInfo.hasAVX2Support() || targetInfo.hasAVX512Support();
  }
  if(feature == "AVX"){
    return mlir::spn::TargetInformation::nativeCPUTarget().hasAVXSupport();
  }
  if(feature == "AVX2"){
    return mlir::spn::TargetInformation::nativeCPUTarget().hasAVX2Support();
  }
  if(feature == "AVX512"){
    return mlir::spn::TargetInformation::nativeCPUTarget().hasAVX512Support();
  }
  // TODO Add query support for more features.
  return false;
}