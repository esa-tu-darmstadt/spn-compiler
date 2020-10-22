//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <spnc.h>
#include <driver/toolchain/CPUToolchain.h>
#include <driver/toolchain/MLIRToolchain.h>
#include <driver/Options.h>
#include <driver/GlobalOptions.h>
#include <util/Logging.h>

using namespace spnc;

Kernel spn_compiler::parseJSON(const std::string& inputFile, const options_t& options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");
  interface::Options::dump();
  auto config = interface::Options::parse(options);
  std::unique_ptr<Job<ModuleOp>> job;
  job = MLIRToolchain::constructJobFromFile(inputFile, *config);
  auto& module = job->execute();
  auto kernel = Kernel("foo", "bar");
  SPDLOG_INFO("Generated Kernel in {}, kernel name {}", kernel.fileName(), kernel.kernelName());
  return kernel;
}

Kernel spn_compiler::parseJSONString(const std::string& jsonString, const options_t& options) {
  SPDLOG_INFO("Welcome to the SPN compiler!");
  interface::Options::dump();
  auto config = interface::Options::parse(options);
  std::unique_ptr<Job<mlir::ModuleOp>> job;
  job = MLIRToolchain::constructJobFromString(jsonString, *config);
  auto& module = job->execute();
  auto kernel = Kernel("foo", "bar");
  SPDLOG_INFO("Generated Kernel in {}, kernel name '{}'", kernel.fileName(), kernel.kernelName());
  return kernel;
}


