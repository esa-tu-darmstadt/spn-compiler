//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPCONVERSION_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPCONVERSION_H

#include "SLPNode.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        enum CreationMode {
          Default,
          Constant,
          Splat,
          BroadcastInsert,
          ConsecutiveLoad,
          Skip
        };

        class ConversionState {

        public:

          explicit ConversionState(SLPNode* root);

          Value getInsertionPoint(NodeVector* vector) const;
          bool isConverted(NodeVector* vector) const;

          void update(NodeVector* vector, Value const& operation, CreationMode const& mode);
          void markSkipped(NodeVector* vector);

          Value getValue(NodeVector* vector) const;
          CreationMode getCreationMode(NodeVector* vector) const;
          Optional<Value> getEarliestEscapingUse(NodeVector* vector, size_t lane) const;

        private:

          struct NodeVectorData {

            /// The operation that was created for this node vector.
            Optional<Value> operation{None};
            /// The way it was created.
            Optional<CreationMode> mode{None};
            /// The earliest (i.e. smallest Loc) escaping use for each lane.
            DenseMap<size_t, Value> earliestEscapingUses;
          };

          DenseMap<NodeVector*, NodeVectorData> vectorData;
          Value earliestInsertionPoint = nullptr;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPCONVERSION_H
