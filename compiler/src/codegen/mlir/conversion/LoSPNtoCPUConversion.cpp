//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPUConversion.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#include "LoSPNtoCPU/Vectorization/VectorOptimizationPasses.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"

void spnc::LoSPNtoCPUConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  bool vectorize = spnc::option::cpuVectorize.get(*this->config);
  pm->addPass(mlir::spn::createLoSPNtoCPUStructureConversionPass(vectorize));
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
