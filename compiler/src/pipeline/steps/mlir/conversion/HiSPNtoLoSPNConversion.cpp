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
//#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "option/GlobalOptions.h"

void spnc::HiSPNtoLoSPNConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  auto config = getContext()->get<Configuration>();
  auto useLogSpace = spnc::option::logSpace.get(*config);
  auto useOptimalRepresentation = spnc::option::optRepresentation.get(*config);
  pm->addPass(mlir::spn::createHiSPNtoLoSPNNodeConversionPass(useLogSpace, useOptimalRepresentation));
  pm->addPass(mlir::spn::createHiSPNtoLoSPNQueryConversionPass(useLogSpace, useOptimalRepresentation));
  auto collectGraphStats = spnc::option::collectGraphStats.get(*config);
  if (collectGraphStats) {
    auto graphStatsFile = spnc::option::graphStatsFile.get(*config);
    pm->addPass(mlir::spn::low::createLoSPNGraphStatsCollectionPass(graphStatsFile));
  }
}