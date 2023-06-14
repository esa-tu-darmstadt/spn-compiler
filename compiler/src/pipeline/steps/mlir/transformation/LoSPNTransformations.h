//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

namespace spnc {

  ///
  /// MLIR pass pipeline performing dialect-internal transformations on the LoSPN dialect.
  class LoSPNTransformations : public MLIRPassPipeline<LoSPNTransformations> {

  public:

    using MLIRPassPipeline<LoSPNTransformations>::MLIRPassPipeline;

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

    void preProcess(mlir::ModuleOp* inputModule) override;

    STEP_NAME("lospn-transform")

  private:

    std::string translateType(mlir::Type type);

  };
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H
