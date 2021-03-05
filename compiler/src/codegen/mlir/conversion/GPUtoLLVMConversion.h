//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H

#include "../MLIRPassPipeline.h"

namespace spnc {

  ///
  /// Action performing a series of transformations on an MLIR module
  /// to lower from GPU (and other dialects) to LLVM dialect.
  struct GPUtoLLVMConversion : public MLIRPipelineBase<GPUtoLLVMConversion> {

    using MLIRPipelineBase<GPUtoLLVMConversion>::MLIRPipelineBase;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_GPUTOLLVMCONVERSION_H
