//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

#include "mlir/Pass/PassOptions.h"
#include "llvm/Frontend/Driver/CodeGenOptions.h"

namespace mlir {
namespace spn {

struct LoSPNtoCPUPipelineOptions
    : public mlir::PassPipelineOptions<LoSPNtoCPUPipelineOptions> {
  PassOptions::Option<bool> vectorize{
      *this, "vectorize",
      llvm::cl::desc("Vectorize code generated for CPU targets."),
      llvm::cl::init(false)};

  PassOptions::Option<bool> replaceGatherWithShuffle{
      *this, "use-shuffle",
      llvm::cl::desc("Optimize gather loads into vector loads and shuffles"),
      llvm::cl::init(false)};
};

void registerLoSPNtoCPUPipeline();
LogicalResult buildLoSPNtoCPUPipeline(mlir::OpPassManager &pm,
                                      const LoSPNtoCPUPipelineOptions &options);

} // namespace spn
} // namespace mlir