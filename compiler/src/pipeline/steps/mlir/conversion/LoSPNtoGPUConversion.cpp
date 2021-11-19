//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoGPUConversion.h"
#include "LoSPNtoGPU/LoSPNtoGPUConversionPasses.h"
#include "LoSPNtoGPU/LoSPNtoGPUPasses.h"
#include "mlir/InitAllPasses.h"
#include <option/GlobalOptions.h>

void spnc::LoSPNtoGPUConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->addPass(mlir::spn::createLoSPNtoGPUStructureConversionPass());
  pm->addPass(mlir::spn::createGPUCopyEliminationPass());
  pm->addPass(mlir::createGpuKernelOutliningPass());
  auto* config = getContext()->get<Configuration>();
  if (spnc::option::gpuSharedMem.get(*config)) {
    // Add the pass transforming accesses to global memory with
    // preloads to shared memory depending on option value.
    pm->addPass(mlir::spn::createLoSPNGPUSharedMemoryInsertionPass());
  }
  pm->nest<mlir::FuncOp>().addPass(mlir::spn::createGPUBufferDeallocationPass());
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
}