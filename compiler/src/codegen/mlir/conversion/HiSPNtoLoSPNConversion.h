//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_HISPNTOLOSPNCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_HISPNTOLOSPNCONVERSION_H

#include "../MLIRPassPipeline.h"

namespace spnc {

  struct HiSPNtoLoSPNConversion : public MLIRPipelineBase<HiSPNtoLoSPNConversion> {

    using MLIRPipelineBase<HiSPNtoLoSPNConversion>::MLIRPipelineBase;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_HISPNTOLOSPNCONVERSION_H
