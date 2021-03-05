//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoGPUConversion.h"
#include "LoSPNtoGPU/LoSPNtoGPUConversionPasses.h"
#include "mlir/InitAllPasses.h"

void spnc::LoSPNtoGPUConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->addPass(mlir::spn::createLoSPNtoGPUStructureConversionPass());
  pm->addPass(mlir::createGpuKernelOutliningPass());
  pm->addPass(mlir::spn::createLoSPNtoGPUNodeConversionPass());
}