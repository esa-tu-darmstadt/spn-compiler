//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_IPUTOOLCHAIN_H
#define SPNC_COMPILER_SRC_DRIVER_TOOLCHAIN_IPUTOOLCHAIN_H

#include "Kernel.h"
#include "MLIRToolchain.h"
#include "pipeline/Pipeline.h"

namespace spnc {
///
/// Toolchain generating code for CPUs using LLVM.
class IPUToolchain : MLIRToolchain {

public:
  /// Construct a job reading the SPN from an input file.
  /// \param inputFile Input file.
  /// \return Job containing all necessary actions.
  static std::unique_ptr<Pipeline<std::unique_ptr<Kernel>>>
  setupPipeline(const std::string &inputFile);
};
} // namespace spnc

#endif