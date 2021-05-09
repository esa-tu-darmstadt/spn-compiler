//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H

#include "SLPGraph.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        enum ElementFlag {
          KeepNone,
          KeepFirst,
          KeepAll,
          NoExtract
        };

        class ConversionManager {

        public:

          explicit ConversionManager(ArrayRef<SLPNode const*> const& nodes);

          void setInsertionPointFor(NodeVector* vector, PatternRewriter& rewriter) const;
          bool wasConverted(NodeVector* vector) const;

          void update(NodeVector* vector, Value const& operation, ElementFlag const& flag);

          Value getValue(NodeVector* vector) const;
          ElementFlag getElementFlag(NodeVector* vector) const;

          bool hasEscapingUsers(Value const& value) const;
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

          /// true = insert before, false = insert after
          Optional<std::pair<Operation*, bool>> insertionPoint = None;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
