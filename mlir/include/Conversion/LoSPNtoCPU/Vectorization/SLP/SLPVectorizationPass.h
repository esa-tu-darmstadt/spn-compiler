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
    namespace low {
      namespace slp {

        namespace {
          typedef std::pair<Operation*, size_t> Extraction;
        }

        struct SLPVectorizationPass : public PassWrapper<SLPVectorizationPass, OperationPass<FuncOp>> {

        public:

        protected:
          void runOnOperation() override;

        private:

          void transform(SLPGraph const& graph);
          Operation* transform(SLPNode const& node, size_t vectorIndex, size_t spill);
          void updateExtractions(SLPNode const& node, size_t const& vectorIndex, Operation* vectorOp);

          /// Stores where operations can find their operands after vectorization in case their defining operations
          /// were deleted during vectorization.
          std::map<Operation*, std::map<size_t, Extraction>> extractions;

        };
      }
    }
    std::unique_ptr<Pass> createSLPVectorizationPass();
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H
