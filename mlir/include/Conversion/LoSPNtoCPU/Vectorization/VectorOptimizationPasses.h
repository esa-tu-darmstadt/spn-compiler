//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTOROPTIMIZATIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTOROPTIMIZATIONPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
  namespace spn {

    struct ReplaceGatherWithShufflePass : public PassWrapper<ReplaceGatherWithShufflePass, OperationPass<mlir::ModuleOp>> {
    protected:
      void runOnOperation() override;

    };

    std::unique_ptr<mlir::Pass> createReplaceGatherWithShufflePass();

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTOROPTIMIZATIONPASSES_H
