//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <iostream>
#include <driver/toolchain/CPUToolchain.h>

namespace spnc {
    bool spnc::parseJSON(const std::string &inputFile) {
      auto job = CPUToolchain::constructJob(inputFile);
      job->execute();
      return true;
    }
}

