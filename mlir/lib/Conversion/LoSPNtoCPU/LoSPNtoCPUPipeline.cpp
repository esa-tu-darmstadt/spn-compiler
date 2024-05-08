//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/LoSPNtoCPUPipeline.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#include "LoSPNtoCPU/Vectorization/VectorOptimizationPasses.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace spn {
LogicalResult
buildLoSPNtoCPUPipeline(mlir::OpPassManager &pm,
                        const LoSPNtoCPUPipelineOptions &options) {
  LoSPNtoCPUStructureConversionPassOptions structConvOptions;
  structConvOptions.vectorize = options.vectorize;
  structConvOptions.vectorWidth = options.vectorWidth;
  structConvOptions.maxAttempts = options.slpMaxAttempts;
  structConvOptions.maxSuccessfulIterations =
      options.slpMaxSuccessfulIterations;
  structConvOptions.maxNodeSize = options.slpMaxNodeSize;
  structConvOptions.maxLookAhead = options.slpMaxLookAhead;
  structConvOptions.reorderInstructionsDFS = options.slpReorderInstructionsDFS;
  structConvOptions.allowDuplicateElements = options.slpAllowDuplicateElements;
  structConvOptions.allowTopologicalMixing = options.slpAllowTopologicalMixing;
  structConvOptions.useXorChains = options.slpUseXorChains;

  pm.addPass(
      mlir::spn::createLoSPNtoCPUStructureConversionPass(structConvOptions));
  if (options.vectorize) {
    if (options.replaceGatherWithShuffle) {
      pm.addPass(mlir::spn::createReplaceGatherWithShufflePass());
    }
    pm.addPass(mlir::spn::createLoSPNNodeVectorizationPass());
    if (options.replaceGatherWithShuffle) {
      // We need another run of the canonicalizer here to remove
      // lo_spn.to_scalar operations introduced by the replacement of gathers
      // and that should be obsolete after the node vectorization.
      pm.addPass(mlir::createCanonicalizerPass());
    }
  }
  pm.addPass(mlir::spn::createLoSPNtoCPUNodeConversionPass());

  // The remaining bufferization, buffer deallocation and copy removal passes
  // currently need to be placed at this point in the pipeline, as they operate
  // on FuncOp (not SPNKernel/SPNTask) and can therefore only run after the
  // conversion to FuncOp. This could be avoided at least for Kernels by
  // converting them to FuncOp earlier in the pipeline, e.g., during
  // bufferization of Kernels.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tensor::createTensorBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferDeallocationPass());

  return success();
}

void registerLoSPNtoCPUPipeline() {
  mlir::PassPipelineRegistration<LoSPNtoCPUPipelineOptions>(
      "lospn-to-cpu-pipeline",
      "The default pipeline for lowering LoSPN dialect "
      "to a CPU compatible LLVM dialect.",
      buildLoSPNtoCPUPipeline);
}
} // namespace spn
} // namespace mlir