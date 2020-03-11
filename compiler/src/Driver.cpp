//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <iostream>
#include <driver/toolchain/CPUToolchain.h>
#include <driver/Options.h>
#include <util/Logging.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/cfg/env.h>


namespace spnc {

  Kernel spn_compiler::parseJSON(const std::string& inputFile, const options_t& options) {
    initLogger();
    SPDLOG_INFO("Welcome to the SPN compiler!");
    interface::Options::dump();
    auto config = interface::Options::parse(options);
    auto job = CPUToolchain::constructJobFromFile(inputFile, *config);
    auto& kernel = job->execute();
    std::cout << "File: " << kernel.fileName() << " Function: " << kernel.kernelName() << std::endl;
    return kernel;
  }

  Kernel spn_compiler::parseJSONString(const std::string& jsonString, const options_t& options) {
    initLogger();
    interface::Options::dump();
    auto config = interface::Options::parse(options);
    auto job = CPUToolchain::constructJobFromString(jsonString, *config);
    auto& kernel = job->execute();
    return kernel;
  }

  void spn_compiler::initLogger() {
    if(!initOnce){
      auto console = spdlog::stdout_color_mt("console");
      spdlog::set_default_logger(console);
      spdlog::cfg::load_env_levels();
      spdlog::set_pattern("[%c] [%s:%#] -- [%l] %v");
      initOnce = true;
    }
  }
}

