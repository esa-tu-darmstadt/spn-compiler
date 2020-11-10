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
using namespace mlir;

namespace spnc {

  ///
  /// Toolchain generating code for CPUs using LLVM.
  class MLIRToolchain {

  public:
    /// Construct a job reading the SPN from an input file.
    /// \param inputFile Input file.
    /// \param config Compilation option configuration.
    /// \return Job containing all necessary actions.
    static std::unique_ptr<Job<Kernel>> constructJobFromFile(const std::string& inputFile,
                                                             const Configuration& config);

    /// Construct a job reading the SPN from an input string.
    /// \param inputString Input string.
    /// \param config Compilation option configuration.
    /// \return Job containing all necessary actions.
    static std::unique_ptr<Job<Kernel>> constructJobFromString(const std::string& inputString,
                                                               const Configuration& config);

  private:
    static std::unique_ptr<Job<Kernel>> constructJob(std::unique_ptr<ActionWithOutput<std::string>> input,
                                                     const Configuration& config);

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_MLIRTOOLCHAIN_H