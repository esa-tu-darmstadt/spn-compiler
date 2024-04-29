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

  /// SLP vectorizer options
  PassOptions::Option<unsigned> slpMaxAttempts{
      *this, "slp-max-attempts",
      llvm::cl::desc("Maximum number of SLP vectorization attempts"),
      llvm::cl::init(1)};
  PassOptions::Option<unsigned> slpMaxSuccessfulIterations{
      *this, "slp-max-successful-iterations",
      llvm::cl::desc("Maximum number of successful SLP vectorization runs to "
                     "be applied to a function"),
      llvm::cl::init(1)};
  PassOptions::Option<unsigned> slpMaxNodeSize{
      *this, "slp-max-node-size",
      llvm::cl::desc("Maximum multinode size during SLP vectorization in terms "
                     "of the number of vectors they may contain"),
      llvm::cl::init(10)};
  PassOptions::Option<unsigned> slpMaxLookAhead{
      *this, "slp-max-look-ahead",
      llvm::cl::desc("Maximum look-ahead depth when reordering multinode "
                     "operands during SLP vectorization"),
      llvm::cl::init(3)};
  PassOptions::Option<bool> slpReorderInstructionsDFS{
      *this, "slp-reorder-instructions-dfs",
      llvm::cl::desc("Flag to indicate if SLP-vectorized instructions should "
                     "be arranged in DFS order (true) or in BFS order "
                     "(false)"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> slpAllowDuplicateElements{
      *this, "slp-allow-duplicate-elements",
      llvm::cl::desc("Flag to indicate whether duplicate elements are allowed "
                     "in vectors during SLP graph building"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> slpAllowTopologicalMixing{
      *this, "slp-allow-topological-mixing",
      llvm::cl::desc("Flag to indicate if elements with different topological "
                     "depths are allowed in vectors during SLP graph "
                     "building"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> slpUseXorChains{
      *this, "slp-use-xor-chains",
      llvm::cl::desc("Flag to indicate if XOR chains should be used to "
                     "compute look-ahead scores instead of Porpodas's "
                     "algorithm"),
      llvm::cl::init(true)};
};

void registerLoSPNtoCPUPipeline();
LogicalResult buildLoSPNtoCPUPipeline(mlir::OpPassManager &pm,
                                      const LoSPNtoCPUPipelineOptions &options);

} // namespace spn
} // namespace mlir