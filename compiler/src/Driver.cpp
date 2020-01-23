//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <iostream>
#include <driver/toolchain/CPUToolchain.h>

namespace spnc {
    Kernel spn_compiler::parseJSON(const std::string &inputFile) {
      auto job = CPUToolchain::constructJobFromFile(inputFile);
      auto& kernel = job->execute();
      std::cout << "File: " << kernel.fileName() << " Function: " << kernel.kernelName() << std::endl;
      return kernel;
    }

    Kernel spn_compiler::parseJSONString(const std::string &jsonString) {
      auto job = CPUToolchain::constructJobFromString(jsonString);
      auto& kernel = job->execute();
      return kernel;
    }
}

