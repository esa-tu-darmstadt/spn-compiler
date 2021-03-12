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
  pm->addPass(mlir::spn::createLoSPNGPUSharedMemoryInsertionPass());
  pm->addPass(mlir::spn::createLoSPNtoGPUNodeConversionPass());
  // The remaining bufferization, buffer deallocation and copy removal passes
  // currently need to be placed at this point in the pipeline, as they operate
  // on FuncOp (not SPNKernel/SPNTask) and can therefore only run after the
  // conversion to FuncOp. This could be avoided at least for Kernels by
  // converting them to FuncOp earlier in the pipeline, e.g., during
  // bufferization of Kernels.
  pm->nest<mlir::FuncOp>().addPass(mlir::createTensorBufferizePass());
  pm->nest<mlir::FuncOp>().addPass(mlir::createFinalizingBufferizePass());
  pm->nest<mlir::FuncOp>().addPass(mlir::createBufferDeallocationPass());
  pm->addPass(mlir::createCopyRemovalPass());
}