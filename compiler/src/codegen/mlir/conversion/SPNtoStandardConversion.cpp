//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNtoStandardConversion.h"
#include "SPNtoStandard/SPNtoStandardConversionPass.h"

void spnc::SPNtoStandardConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->addPass(mlir::spn::createSPNtoStandardConversionPass());
}
