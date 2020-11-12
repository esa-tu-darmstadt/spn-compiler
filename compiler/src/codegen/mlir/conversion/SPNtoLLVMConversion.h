//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_SPNTOLLVMCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_SPNTOLLVMCONVERSION_H

#include "../MLIRPassPipeline.h"

namespace spnc {

  ///
  /// Action performing a conversion from SPN & Standard dialect to LLVM dialect.
  struct SPNtoLLVMConversion : public MLIRPipelineBase<SPNtoLLVMConversion> {
    using MLIRPipelineBase<SPNtoLLVMConversion>::MLIRPipelineBase;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_SPNTOLLVMCONVERSION_H
