//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_PIPELINE_SPNDIALECTPIPELINE_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_PIPELINE_SPNDIALECTPIPELINE_H

#include "../MLIRPassPipeline.h"

namespace spnc {

  ///
  /// Action running a series of MLIR passes on the SPN dialect.
  class SPNDialectPipeline : public MLIRPipelineBase<SPNDialectPipeline> {

  public:

    using MLIRPipelineBase<SPNDialectPipeline>::MLIRPipelineBase;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_PIPELINE_SPNDIALECTPIPELINE_H
