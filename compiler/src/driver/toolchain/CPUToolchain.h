//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_CPUTOOLCHAIN_H
#define SPNC_CPUTOOLCHAIN_H

#include <llvm/IR/Module.h>
#include <driver/Job.h>
#include <driver/BaseActions.h>
#include <driver/Options.h>
#include <Kernel.h>

using namespace spnc::interface;

namespace spnc {

  ///
  /// Toolchain generating code for CPUs using LLVM.
  /// The generated Kernel iterates over a batch of queries and computes the joint probability.
  class CPUToolchain {

  public:

    /// Construct a job reading the SPN from an input file.
    /// \param inputFile Input file.
    /// \param config Compilation option configuration.
    /// \return Job containing all necessary actions.
    static std::unique_ptr<Job<Kernel>> constructJobFromFile(const std::string& inputFile, const Configuration& config);

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

#endif //SPNC_CPUTOOLCHAIN_H
