//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_GPUTOOLCHAIN_H
#define SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_GPUTOOLCHAIN_H

#include "MLIRToolchain.h"

namespace spnc {

  ///
  /// Toolchain generating code for CUDA GPUs.
  class CUDAGPUToolchain : public MLIRToolchain {

  public:
    /// Construct a job reading the SPN from an input file.
    /// \param inputFile Input file.
    /// \param config Compilation option configuration.
    /// \return Job containing all necessary actions.
    static std::unique_ptr<Job<Kernel>> constructJobFromFile(const std::string& inputFile,
                                                             std::shared_ptr<interface::Configuration> config);

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_GPUTOOLCHAIN_H
