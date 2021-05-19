//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SEEDING_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SEEDING_H

#include "mlir/IR/Operation.h"
#include "SLPGraph.h"
#include <unordered_set>

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SeedAnalysis {

        public:

          SeedAnalysis(Operation* rootOp, unsigned width);

          void markAllUnavailable(ValueVector* root);

          virtual SmallVector<Value, 4> next() const = 0;

        protected:
          Operation* const rootOp;
          unsigned const width;
          std::unordered_set<Operation*> availableOps;
        };

        class TopDownAnalysis : public SeedAnalysis {
        public:
          TopDownAnalysis(Operation* rootOp, unsigned width);
          SmallVector<Value, 4> next() const override;
        private:
          DenseMap<Value, unsigned> computeOpDepths() const;
        };

        class BottomUpAnalysis : public SeedAnalysis {
        public:
          BottomUpAnalysis(Operation* rootOp, unsigned width);
          SmallVector<Value, 4> next() const override;
        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SEEDING_H
