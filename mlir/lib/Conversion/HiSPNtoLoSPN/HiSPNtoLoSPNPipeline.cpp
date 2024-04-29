//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPNtoLoSPN/HiSPNtoLoSPNPipeline.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "LoSPN/LoSPNPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace spn {
LogicalResult
buildHiSPNtoLoSPNPipeline(mlir::OpPassManager &pm,
                          const HiSPNtoLoSPNPipelineOptions &options) {
  HiSPNtoLoSPNNodeConversionPassOptions nodeOptions;
  nodeOptions.computeLogSpace = options.computeLogSpace;
  nodeOptions.optimizeRepresentation = options.optimizeRepresentation;
  pm.addPass(mlir::spn::createHiSPNtoLoSPNNodeConversionPass(nodeOptions));

  HiSPNtoLoSPNQueryConversionPassOptions queryOptions;
  queryOptions.computeLogSpace = options.computeLogSpace;
  queryOptions.optimizeRepresentation = options.optimizeRepresentation;
  pm.addPass(mlir::spn::createHiSPNtoLoSPNQueryConversionPass(queryOptions));

  if (options.collectGraphStats) {
    pm.addPass(mlir::spn::low::createLoSPNGraphStatsCollectionPass(
        options.graphStatsFile));
  }
  return success();
}

void registerHiSPNtoLoSPNPipeline() {
  mlir::PassPipelineRegistration<HiSPNtoLoSPNPipelineOptions>(
      "hispn-to-lospn-pipeline",
      "The default pipeline for lowering HiSPN dialect "
      "to the LoSPN dialect",
      buildHiSPNtoLoSPNPipeline);
}
} // namespace spn
} // namespace mlir