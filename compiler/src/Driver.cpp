//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <spnc.h>
#include <driver/toolchain/CPUToolchain.h>
#include <driver/Options.h>
#include <driver/GlobalOptions.h>
#include <util/Logging.h>
#if SPNC_CUDA_SUPPORT
// Only include if CUDA GPU support was enabled.
#include <driver/toolchain/CUDAGPUToolchain.h>
#endif

using namespace spnc;

Kernel spn_compiler::compileQuery(const std::string& inputFile, const options_t& options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");
  interface::Options::dump();
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


