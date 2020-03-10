//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <iostream>
#include <driver/toolchain/CPUToolchain.h>
#include <driver/Options.h>

namespace spnc {

  Kernel spn_compiler::parseJSON(const std::string& inputFile, const options_t& options) {
    interface::Options::dump();
    auto config = interface::Options::parse(options);
    auto job = CPUToolchain::constructJobFromFile(inputFile, *config);
    auto& kernel = job->execute();
    std::cout << "File: " << kernel.fileName() << " Function: " << kernel.kernelName() << std::endl;
    return kernel;
  }

  Kernel spn_compiler::parseJSONString(const std::string& jsonString, const options_t& options) {
    interface::Options::dump();
    auto config = interface::Options::parse(options);
    auto job = CPUToolchain::constructJobFromString(jsonString, *config);
    auto& kernel = job->execute();
    return kernel;
  }
}

