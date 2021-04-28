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

        class ConversionManager {

        public:

          explicit ConversionManager(SLPNode* root);

          Value getInsertionPoint(NodeVector* vector) const;
          bool isConverted(NodeVector* vector) const;

          void update(NodeVector* vector, Value const& operation, CreationMode const& mode);
          void markSkipped(NodeVector* vector);

          Value getValue(NodeVector* vector) const;
          CreationMode getCreationMode(NodeVector* vector) const;

          bool hasEscapingUsers(Value const& value) const;
          Operation* moveEscapingUsersBehind(NodeVector* vector, Value const& operation) const;

        private:

          struct NodeVectorData {
            /// The operation that was created for this node vector.
            Optional<Value> operation{None};
            /// The way it was created.
            Optional<CreationMode> mode{None};
          };
          DenseMap<NodeVector*, NodeVectorData> vectorData;

          /// Stores escaping users for each value.
          DenseMap<Value, SmallVector<Operation*, 2>> escapingUsers;

          Value latestInsertion;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPCONVERSION_H
