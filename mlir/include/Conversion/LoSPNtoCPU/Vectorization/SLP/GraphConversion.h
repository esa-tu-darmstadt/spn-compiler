//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H

#include "SLPGraph.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        enum ElementFlag {
          KeepNone,
          KeepFirst,
          KeepAll,
          NoExtract,
          Skip
        };

        class ConversionManager {

        public:

          explicit ConversionManager(SLPNode* root);

          Value getInsertionPoint(NodeVector* vector) const;
          bool wasConverted(NodeVector* vector) const;

          void update(NodeVector* vector, Value const& operation, ElementFlag const& flag);
          void markSkipped(NodeVector* vector);

          Value getValue(NodeVector* vector) const;
          ElementFlag getElementFlag(NodeVector* vector) const;

          bool hasEscapingUsers(Value const& value) const;
          /// Recursively moves all users (and their users and so on) of the vector behind it.
          void recursivelyMoveUsersAfter(NodeVector* vector) const;
          Operation* getEarliestEscapingUser(Value const& value) const;

        private:

          struct NodeVectorData {
            /// The operation that was created for this node vector.
            Optional<Value> operation{None};
            /// The way it was created.
            Optional<ElementFlag> flag{None};
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

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
