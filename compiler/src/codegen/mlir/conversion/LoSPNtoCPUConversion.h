//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_LOSPNTOCPUCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_LOSPNTOCPUCONVERSION_H

#include "driver/Job.h"
#include "../MLIRPassPipeline.h"

namespace spnc {

  struct LoSPNtoCPUConversion : public MLIRPipelineBase<LoSPNtoCPUConversion> {

  public:

    LoSPNtoCPUConversion(ActionWithOutput<mlir::ModuleOp>& _input,
                         std::shared_ptr<mlir::MLIRContext> ctx,
                         std::shared_ptr<mlir::ScopedDiagnosticHandler> handler,
                         std::shared_ptr<KernelInfo> kernelInformation) :
        MLIRPipelineBase<LoSPNtoCPUConversion>(_input, std::move(ctx), std::move(handler)),
        kernelInfo{std::move(kernelInformation)} {}

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

  private:

    std::shared_ptr<KernelInfo> kernelInfo;

    using MLIRPipelineBase<LoSPNtoCPUConversion>::MLIRPipelineBase;

  };
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_LOSPNTOCPUCONVERSION_H
