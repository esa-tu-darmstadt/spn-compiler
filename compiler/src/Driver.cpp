//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <spnc.h>
#include <iostream>
#include <driver/toolchain/CPUToolchain.h>
#include <driver/Options.h>
#include <util/Logging.h>

using namespace spnc;

Kernel spn_compiler::parseJSON(const std::string& inputFile, const options_t& options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");
  interface::Options::dump();
  auto config = interface::Options::parse(options);
  auto job = CPUToolchain::constructJobFromFile(inputFile, *config);
  auto& kernel = job->execute();
  SPDLOG_INFO("Generated Kernel in {}, kernel name {}", kernel.fileName(), kernel.kernelName());
  return kernel;
}

Kernel spn_compiler::parseJSONString(const std::string& jsonString, const options_t& options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");
  interface::Options::dump();
  auto config = interface::Options::parse(options);
  auto job = CPUToolchain::constructJobFromString(jsonString, *config);
  auto& kernel = job->execute();
  SPDLOG_INFO("Generated Kernel in {}, kernel name '{}'", kernel.fileName(), kernel.kernelName());
  return kernel;
}


