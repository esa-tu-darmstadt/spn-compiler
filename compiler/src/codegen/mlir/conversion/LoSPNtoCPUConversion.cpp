//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPUConversion.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"

spnc::LoSPNtoCPUConversion::LoSPNtoCPUConversion(ActionWithOutput<mlir::ModuleOp>& _input,
                                                 std::shared_ptr<mlir::MLIRContext> ctx,
                                                 std::shared_ptr<mlir::ScopedDiagnosticHandler> handler,
                                                 bool enableVectorization) :
    MLIRPipelineBase<LoSPNtoCPUConversion>(_input,
                                           std::move(ctx),
                                           std::move(handler)),
    vectorize{enableVectorization} {}

void spnc::LoSPNtoCPUConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->enableVerifier(false);
  pm->enableIRPrinting();
  pm->addPass(mlir::spn::createLoSPNtoCPUStructureConversionPass(vectorize));
  if (vectorize) {
    pm->addPass(mlir::spn::createLoSPNNodeVectorizationPass());
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