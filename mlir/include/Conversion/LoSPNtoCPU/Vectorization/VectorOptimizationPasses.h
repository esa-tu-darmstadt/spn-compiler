//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTOROPTIMIZATIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTOROPTIMIZATIONPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    struct ReplaceGatherWithShufflePass : public PassWrapper<ReplaceGatherWithShufflePass, OperationPass<ModuleOp>> {
    protected:
      void runOnOperation() override;

    };

    std::unique_ptr<mlir::Pass> createReplaceGatherWithShufflePass();

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTOROPTIMIZATIONPASSES_H
