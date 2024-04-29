//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "HiSPNtoLoSPNConversion.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNPasses.h"

void spnc::HiSPNtoLoSPNConversion::initializePassPipeline(
    mlir::PassManager *pm, mlir::MLIRContext *ctx) {
  bool useLogSpace = spnc::option::logSpace;
  bool useOptimalRepresentation = spnc::option::optRepresentation;
  // TODO: Refactor these passes into a pipeline. Make the options
  // pipeline-specific.
  pm->addPass(mlir::spn::createHiSPNtoLoSPNNodeConversionPass(
      useLogSpace, useOptimalRepresentation));
  pm->addPass(mlir::spn::createHiSPNtoLoSPNQueryConversionPass(
      useLogSpace, useOptimalRepresentation));
  if (spnc::option::collectGraphStats) {
    pm->addPass(mlir::spn::low::createLoSPNGraphStatsCollectionPass(
        spnc::option::graphStatsFile));
  }
}