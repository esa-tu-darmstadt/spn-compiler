//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNtoLLVMConversion.h"
#include "SPNtoLLVM/SPNtoLLVMConversionPass.h"

void spnc::SPNtoLLVMConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->addPass(mlir::spn::createSPNtoLLVMConversionPass());
}
