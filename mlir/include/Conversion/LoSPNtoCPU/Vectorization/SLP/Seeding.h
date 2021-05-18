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

        enum Order {
          ///
          DefUse,
          ///
          UseDef,
          /// TODO?
          Chain
        };

        class SeedAnalysis {

        public:

          SeedAnalysis(Operation* rootOp, unsigned width);

          SmallVector<Value, 4> next(Order const& mode);
          void markAllUnavailable(ValueVector* root);

        private:
          Operation* rootOp;
          unsigned const width;
          std::unordered_set<Operation*> availableOps;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SEEDING_H
