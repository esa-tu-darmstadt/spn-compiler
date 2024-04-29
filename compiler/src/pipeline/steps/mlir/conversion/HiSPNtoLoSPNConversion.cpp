//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPNtoLoSPNConversion.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNPipeline.h"
#include "option/Options.h"

void spnc::HiSPNtoLoSPNConversion::initializePassPipeline(
    mlir::PassManager *pm, mlir::MLIRContext *ctx) {

  mlir::spn::HiSPNtoLoSPNPipelineOptions options;
  options.computeLogSpace = spnc::option::logSpace.getValue();
  options.optimizeRepresentation = spnc::option::optRepresentation.getValue();
  options.collectGraphStats = spnc::option::collectGraphStats.getValue();

  if (mlir::spn::buildHiSPNtoLoSPNPipeline(*pm, options).failed()) {
    llvm::errs() << "Failed to build HiSPN to LoSPN pipeline\n";
  }
}