//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_LOSPNTOGPUCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_LOSPNTOGPUCONVERSION_H

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

namespace spnc {

  ///
  /// MLIR pass pipeline to lower from the LoSPN dialect to a combination of
  /// upstream dialects when targeting GPUS.
  struct LoSPNtoGPUConversion : public MLIRPassPipeline<LoSPNtoGPUConversion> {

    using MLIRPassPipeline<LoSPNtoGPUConversion>::MLIRPassPipeline;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

    STEP_NAME("lospn-to-gpu")
  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_LOSPNTOGPUCONVERSION_H
