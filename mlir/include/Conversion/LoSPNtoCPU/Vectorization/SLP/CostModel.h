//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H

#include "SLPNode.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        // TODO: google "use weak_ptrs as cache"

        class CostModel {
        public:
          unsigned getScalarCost(Value const& value);
          unsigned getVectorCost(NodeVector const* vector);
        protected:
          virtual unsigned computeScalarCost(Value const& value) = 0;
          virtual unsigned computeVectorCost(NodeVector const* vector) = 0;
        private:
          DenseMap<Value, unsigned> cachedScalarCosts;
          DenseMap<NodeVector const*, unsigned> cachedVectorCosts;
        };

        class UnitCostModel : public CostModel {
          unsigned computeScalarCost(Value const& value) override;
          unsigned computeVectorCost(NodeVector const* vector) override;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H
