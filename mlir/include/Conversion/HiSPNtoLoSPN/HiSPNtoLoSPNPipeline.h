//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace spn {

struct HiSPNtoLoSPNPipelineOptions
    : public mlir::PassPipelineOptions<HiSPNtoLoSPNPipelineOptions> {
  PassOptions::Option<bool> computeLogSpace{
      *this, "use-log-space",
      llvm::cl::desc("Compute in log-space instead of linear space"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> optimizeRepresentation{
      *this, "optimize-representation",
      llvm::cl::desc("Optimize representation for computation"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> collectGraphStats{
      *this, "collect-graph-stats",
      llvm::cl::desc("Inserts a pass that collects static graph statistics "
                     "during compilation"),
      llvm::cl::init(false)};

  PassOptions::Option<std::string> graphStatsFile{
      *this, "spnc-graph-stats-file",
      llvm::cl::desc("Output file for static graph statistics"),
      llvm::cl::init("/tmp/stats.json")};
};

void registerHiSPNtoLoSPNPipeline();
LogicalResult
buildHiSPNtoLoSPNPipeline(mlir::OpPassManager &pm,
                          const HiSPNtoLoSPNPipelineOptions &options);

} // namespace spn
} // namespace mlir