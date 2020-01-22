//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <iostream>
#include <driver/toolchain/CPUToolchain.h>

namespace spnc {
    bool spn_compiler::parseJSON(const std::string &inputFile) {
      auto job = CPUToolchain::constructJobFromFile(inputFile);
      job->execute();
      return true;
    }

    bool spn_compiler::parseJSONString(const std::string &jsonString) {
      std::cout << jsonString << std::endl;
      return true;
    }
}

