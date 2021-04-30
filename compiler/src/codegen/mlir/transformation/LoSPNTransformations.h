//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H

#include <driver/Job.h>
#include "../MLIRPassPipeline.h"

namespace spnc {

  ///
  /// Action performing dialect-internal transformations on the LoSPN dialect.
  class LoSPNTransformations : public MLIRPipelineBase<LoSPNTransformations> {

  public:

    LoSPNTransformations(ActionWithOutput<mlir::ModuleOp>& _input,
                         std::shared_ptr<mlir::MLIRContext> ctx,
                         std::shared_ptr<mlir::ScopedDiagnosticHandler> handler,
                         std::shared_ptr<KernelInfo> kernelInformation) :
        MLIRPipelineBase<LoSPNTransformations>(_input, std::move(ctx), std::move(handler)),
        kernelInfo{std::move(kernelInformation)} {}

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

    void preProcess(mlir::ModuleOp* inputModule) override;

  private:

    std::string translateType(mlir::Type type);

    std::shared_ptr<KernelInfo> kernelInfo;

  };
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORMATION_LOSPNTRANSFORMATIONS_H
