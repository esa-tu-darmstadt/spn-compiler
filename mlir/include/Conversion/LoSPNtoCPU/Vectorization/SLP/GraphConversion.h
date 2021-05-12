//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H

#include "SLPGraph.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        enum ElementFlag {
          /// Erase all elements after conversion.
          KeepNone,
          /// Erase all but the first element after conversion.
          KeepFirst,
          /// Do not erase any element after conversion.
          KeepAll,
          /// Erase all elements, but also do not create extract operations for them.
          KeepNoneNoExtract
        };

        class ConversionManager {

        public:

          ConversionManager(SLPNode* root, PatternRewriter& rewriter);

          void setInsertionPointFor(NodeVector* vector) const;
          bool wasConverted(NodeVector* vector) const;

          void update(NodeVector* vector, Value const& operation, ElementFlag const& flag);

          Value getValue(NodeVector* vector) const;
          ElementFlag getElementFlag(NodeVector* vector) const;

          ArrayRef<NodeVector*> conversionOrder() const;

          bool hasEscapingUsers(Value const& value) const;
          Operation* getEarliestEscapingUser(Value const& value) const;

          Value getConstant(Location const& loc, Attribute const& attribute);

        private:

          SmallVector<NodeVector*> order;

          struct NodeVectorData {
            /// The operation that was created for this node vector.
            Optional<Value> operation{None};
            /// What to do with its elements after conversion.
            Optional<ElementFlag> flag{None};
          };
          DenseMap<NodeVector*, NodeVectorData> vectorData;

          /// Stores escaping users for each value.
          DenseMap<Value, SmallVector<Operation*, 2>> escapingUsers;

          /// true = insert before, false = insert after
          Optional<std::pair<Operation*, bool>> insertionPoint = None;

          /// For creating constants & setting insertion points.
          PatternRewriter& rewriter;

          /// Helps avoid creating duplicate constants.
          OperationFolder folder;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
