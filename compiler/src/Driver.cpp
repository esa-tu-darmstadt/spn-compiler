//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <iostream>
#include <driver/toolchain/CPUToolchain.h>
#include <driver/toolchain/MLIRToolchain.h>

namespace spnc {
    Kernel spn_compiler::parseJSON(const std::string &inputFile) {
      auto job = MLIRToolchain::constructJobFromFile(inputFile);
      auto& mlir = job->execute();
      mlir.dump();
      Kernel kernel{"/tmp/foo.o", "bar_baz"};
      std::cout << "\nFile: " << kernel.fileName() << " Function: " << kernel.kernelName() << std::endl;
      return kernel;
    }

    Kernel spn_compiler::parseJSONString(const std::string &jsonString) {
      auto job = CPUToolchain::constructJobFromString(jsonString);
      auto& kernel = job->execute();
      return kernel;
    }
}

