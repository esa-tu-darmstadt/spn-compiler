//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "SLPGraph.h"

namespace mlir {
  namespace spn {

    struct SLPVectorizationPass : public PassWrapper<SLPVectorizationPass, OperationPass<FuncOp>> {

    public:

    protected:
      void runOnOperation() override;

    private:

      void transform(SLPGraph& graph);
      Operation* transform(SLPNode& node, bool isRoot);

      std::map<Operation*, std::vector<std::pair<Operation*, size_t>>> extractOps;

    };

    std::unique_ptr<Pass> createSLPVectorizationPass();
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H
