//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_GPUTOOLCHAIN_H
#define SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_GPUTOOLCHAIN_H

#include "MLIRToolchain.h"
#include "driver/pipeline/Pipeline.h"

namespace spnc {

  ///
  /// Toolchain generating code for CUDA GPUs.
  class CUDAGPUToolchain : public MLIRToolchain {

  public:
    /// Construct a job reading the SPN from an input file.
    /// \param inputFile Input file.
    /// \param config Compilation option configuration.
    /// \return Job containing all necessary actions.
    static std::unique_ptr<Pipeline<Kernel>> setupPipeline(const std::string& inputFile,
                                                           std::unique_ptr<interface::Configuration> config);

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_GPUTOOLCHAIN_H
