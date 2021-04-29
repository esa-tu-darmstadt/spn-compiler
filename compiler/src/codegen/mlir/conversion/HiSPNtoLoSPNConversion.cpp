//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "HiSPNtoLoSPNConversion.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNPasses.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "driver/GlobalOptions.h"


void spnc::HiSPNtoLoSPNConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
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