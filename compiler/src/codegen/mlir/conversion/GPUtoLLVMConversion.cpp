//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "GPUtoLLVMConversion.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/InitAllPasses.h"

void spnc::GPUtoLLVMConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->addPass(mlir::createLowerToCFGPass());
  auto& kernelPm = pm->nest<mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(mlir::createStripDebugInfoPass());
  kernelPm.addPass(mlir::createLowerGpuOpsToNVVMOpsPass());
}