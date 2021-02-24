//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H

#include "../MLIRPassPipeline.h"

namespace spnc {

  ///
  /// Action performing dialect-internal transformations on the LoSPN dialect.
  struct LoSPNTransformations : public MLIRPipelineBase<LoSPNTransformations> {
    using MLIRPipelineBase<LoSPNTransformations>::MLIRPipelineBase;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);
  };
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H
