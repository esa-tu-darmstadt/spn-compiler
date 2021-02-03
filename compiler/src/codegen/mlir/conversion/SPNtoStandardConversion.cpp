//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNtoStandardConversion.h"
#include "SPNtoStandard/SPNtoStandardConversionPass.h"

spnc::SPNtoStandardConversion::SPNtoStandardConversion(ActionWithOutput<mlir::ModuleOp>& input,
                                                       std::shared_ptr<mlir::MLIRContext> ctx,
                                                       std::shared_ptr<mlir::ScopedDiagnosticHandler> handler,
                                                       bool cpuVectorize) :
    vectorize{cpuVectorize},
    MLIRPipelineBase<SPNtoStandardConversion>(input, std::move(ctx), std::move(handler)) {}

void spnc::SPNtoStandardConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {
  pm->addPass(mlir::spn::createSPNtoStandardConversionPass(vectorize));
}
