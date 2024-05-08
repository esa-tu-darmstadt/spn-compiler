//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPUConversion.h"
#include "LoSPNtoCPU/LoSPNtoCPUPipeline.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "option/Options.h"

void spnc::LoSPNtoCPUConversion::initializePassPipeline(
    mlir::PassManager *pm, mlir::MLIRContext *ctx) {
  mlir::spn::LoSPNtoCPUPipelineOptions options;
  options.vectorize = option::vectorize.getValue();
  options.replaceGatherWithShuffle =
      option::replaceGatherWithShuffle.getValue();
  options.vectorWidth = option::vectorWidth.getValue();
  options.slpMaxAttempts = option::slpMaxAttempts.getValue();
  options.slpMaxSuccessfulIterations =
      option::slpMaxSuccessfulIterations.getValue();
  options.slpMaxNodeSize = option::slpMaxNodeSize.getValue();
  options.slpMaxLookAhead = option::slpMaxLookAhead.getValue();
  options.slpReorderInstructionsDFS =
      option::slpReorderInstructionsDFS.getValue();
  options.slpAllowDuplicateElements =
      option::slpAllowDuplicateElements.getValue();
  options.slpAllowTopologicalMixing =
      option::slpAllowTopologicalMixing.getValue();
  options.slpUseXorChains = option::slpUseXorChains.getValue();

  mlir::LogicalResult result = mlir::spn::buildLoSPNtoCPUPipeline(*pm, options);
  if (mlir::failed(result)) {
    llvm::errs() << "Failed to add LoSPN to CPU pipeline\n";
  }
}