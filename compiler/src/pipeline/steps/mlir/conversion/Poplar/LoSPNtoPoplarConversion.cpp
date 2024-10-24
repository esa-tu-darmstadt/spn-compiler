//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoPoplarConversion.h"
#include "CPUtoPoplar/CPUtoPoplarConversionPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUPipeline.h"

using namespace mlir;

void spnc::LoSPNtoPoplarConversion::initializePassPipeline(
    mlir::PassManager *pm, mlir::MLIRContext *ctx) {
  // First, use the LoSPN to CPU pipeline. The name is misleading, but the
  // pipeline lowers the LoSPN dialect to standard operations.
  mlir::spn::LoSPNtoCPUPipelineOptions options;
  options.vectorize = option::vectorize.getValue();
  options.replaceGatherWithShuffle =
      option::replaceGatherWithShuffle.getValue();
  options.vectorWidth = option::vectorWidth.getValue();
  options.slpMaxAttempts = option::slpMaxAttempts.getValue();
  options.slpMaxSuccessfulIterations =
      option::slpMaxSuccessfulIterations.getValue();
  options.slpMaxNodeSize = option::slpMaxNodeSize.getValue();
  options.slpMaxLookAhead = option::slpMaxLookAhead.getValue();
  options.slpReorderInstructionsDFS =
      option::slpReorderInstructionsDFS.getValue();
  options.slpAllowDuplicateElements =
      option::slpAllowDuplicateElements.getValue();
  options.slpAllowTopologicalMixing =
      option::slpAllowTopologicalMixing.getValue();
  options.slpUseXorChains = option::slpUseXorChains.getValue();

  mlir::LogicalResult result = mlir::spn::buildLoSPNtoCPUPipeline(*pm, options);
  if (mlir::failed(result)) {
    llvm::errs() << "Failed to add LoSPN to CPU pipeline\n";
  }

  /**
    Input here:
    module {
      func.func @task_0(%arg0: memref<?x10xf64>, %arg1: memref<1x?xf32>) {


        return
      }
      func.func @spn_kernel(%arg0: memref<?x10xf64>, %arg1: memref<1x?xf32>) {
        call @task_0(%arg0, %arg1) : (memref<?x10xf64>, memref<1x?xf32>) -> ()
        return
      }
    }

    1. Turn task into codelet
    2. Turn kernel into graph + program
   */

  pm->addPass(mlir::spn::createCPUtoPoplarConversionPass());
}