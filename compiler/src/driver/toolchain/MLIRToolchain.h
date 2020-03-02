//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H
#define SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H

#include <mlir/IR/Module.h>
#include <driver/Job.h>

namespace spnc {

  class MLIRToolchain {

  public:
    static std::unique_ptr<Job<Kernel>> constructJobFromFile(const std::string& inputFile);

    static std::unique_ptr<Job<Kernel>> constructJobFromString(const std::string& inputString);

  private:
    static std::unique_ptr<Job<Kernel>> constructJob(std::unique_ptr<ActionWithOutput<std::string>> input);

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H
