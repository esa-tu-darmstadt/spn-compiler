//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <spnc.h>
#include <driver/toolchain/CPUToolchain.h>
#include <driver/toolchain/CUDAGPUToolchain.h>
#include <driver/Options.h>
#include <driver/GlobalOptions.h>
#include <util/Logging.h>

using namespace spnc;

Kernel spn_compiler::compileQuery(const std::string& inputFile, const options_t& options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");
  interface::Options::dump();
  auto config = interface::Options::parse(options);
  std::unique_ptr<Job<Kernel>> job;
  if (spnc::option::compilationTarget.get(*config) == option::TargetMachine::CUDA) {
    job = CUDAGPUToolchain::constructJobFromFile(inputFile, config);
  } else {
    job = CPUToolchain::constructJobFromFile(inputFile, config);
  }
  auto kernel = job->execute();
  SPDLOG_INFO("Generated Kernel in {}, kernel name {}", kernel.fileName(), kernel.kernelName());
  return kernel;
}


