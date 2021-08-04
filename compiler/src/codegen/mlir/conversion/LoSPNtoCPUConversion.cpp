//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPUConversion.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#include "LoSPNtoCPU/Vectorization/VectorOptimizationPasses.h"
#include "LoSPN/LoSPNPasses.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include <driver/GlobalOptions.h>
#include <TargetInformation.h>

void spnc::LoSPNtoCPUConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  bool vectorize = spnc::option::cpuVectorize.get(*this->config);
  pm->addPass(mlir::spn::createLoSPNtoCPUStructureConversionPass(
      vectorize,
      spnc::option::slpMaxAttempts.get(*this->config),
      spnc::option::slpMaxSuccessfulIterations.get(*this->config),
      spnc::option::slpMaxNodeSize.get(*this->config),
      spnc::option::slpMaxLookAhead.get(*this->config),
      spnc::option::slpReorderInstructionsDFS.get(*this->config),
      spnc::option::slpAllowDuplicateElements.get(*this->config),
      spnc::option::slpAllowTopologicalMixing.get(*this->config))
  );
  if (vectorize) {
    auto useShuffle = spnc::option::replaceGatherWithShuffle.get(*this->config);
    if (useShuffle) {
      pm->addPass(mlir::spn::createReplaceGatherWithShufflePass());
    }
    pm->addPass(mlir::spn::createLoSPNNodeVectorizationPass());
    if (useShuffle) {
      // We need another run of the canonicalizer here to remove lo_spn.to_scalar
      // operations introduced by the replacement of gathers and that should
      // be obsolete after the node vectorization.
      pm->addPass(mlir::createCanonicalizerPass());
    }
  }
  pm->addPass(mlir::spn::createLoSPNtoCPUNodeConversionPass());
  if (mlir::spn::TargetInformation::nativeCPUTarget().isAARCH64Target() &&
      spnc::option::vectorLibrary.get(*config) == spnc::option::VectorLibrary::ARM) {
    // The ARM Optimized Routines are currently not available through the regular TargetLibraryInfo
    // interface of opt/llc, so replacement with optimized implementations of elementary
    // functions (e.g., exp, log), cannot happen in the backend. Instead, we add our own pass
    // performing the replacement in explicitly defined cases here.
    pm->addPass(mlir::spn::low::createReplaceARMOptimizedRoutinesPass());
  }
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
