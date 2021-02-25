//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "HiSPNtoLoSPNConversion.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"


void spnc::HiSPNtoLoSPNConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->addPass(mlir::spn::createHiSPNtoLoSPNNodeConversionPass());
  pm->addPass(mlir::spn::createHiSPNtoLoSPNQueryConversionPass());
}