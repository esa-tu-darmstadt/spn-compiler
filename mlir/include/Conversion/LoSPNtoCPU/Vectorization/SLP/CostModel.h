//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H

#include "SLPGraph.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        // TODO: google "use weak_ptrs as cache"

        class CostModel {
        public:
          unsigned getScalarCost(Value const& value);
          unsigned getSuperwordCost(Superword const& superword);
        protected:
          virtual unsigned computeScalarCost(Value const& value) = 0;
          virtual unsigned computeSuperwordCost(Superword const& superword) = 0;
        private:
          DenseMap<Value, unsigned> cachedScalarCosts;
          DenseMap<Superword const*, unsigned> cachedSuperwordCosts;
        };

        class UnitCostModel : public CostModel {
          unsigned computeScalarCost(Value const& value) override;
          unsigned computeSuperwordCost(Superword const& superword) override;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H
