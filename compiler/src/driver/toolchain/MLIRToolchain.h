//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H
#define SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H

#include <mlir/IR/Module.h>
#include <driver/Job.h>
#include <driver/Options.h>

using namespace spnc::interface;

namespace spnc {

  class MLIRToolchain {

  public:
    static std::unique_ptr<Job<Kernel>> constructJobFromFile(const std::string& inputFile,
                                                             const Configuration& config);

    static std::unique_ptr<Job<Kernel>> constructJobFromString(const std::string& inputString,
                                                               const Configuration& config);

  private:
    static std::unique_ptr<Job<Kernel>> constructJob(std::unique_ptr<ActionWithOutput<std::string>> input,
                                                     const Configuration& config);

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H
