//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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

          explicit ConversionManager(PatternRewriter& rewriter);

          void initConversion(Superword* root);

          void setInsertionPointFor(Superword* superword) const;
          bool wasConverted(Superword* superword) const;

          void update(Superword* superword, Value const& operation, ElementFlag flag);

          Value getValue(Superword* superword) const;
          ElementFlag getElementFlag(Superword* superword) const;

          ArrayRef<Superword*> conversionOrder() const;

          bool hasEscapingUsers(Value const& value) const;

          Value getOrCreateConstant(Location const& loc, Attribute const& attribute);
          Value getOrExtractValue(Value const& value);
          void createExtractionFor(Value const& value);

        private:

          SmallVector<Superword*> order;

          struct CreationData {
            /// The operation that was created for this superword.
            Value operation;
            /// What to do with its elements after conversion.
            ElementFlag flag;
          };
          DenseMap<Superword*, CreationData> creationData;
          DenseMap<Value, std::pair<Superword*, size_t>> superwordPositions;

          /// Stores insertion points. An entry {k, v} means that superword k will be inserted right after superword v.
          /*
           * NOTE: we cannot use a single value that is updated after every created vector op because this would require
           * Operation::isBeforeInBlock() in case the created vector op was a leaf op and was created *behind* the last
           * vector op (i.e. if lastVectorOp.isBeforeInBlock(createdVectorOp) then lastVectorOp = createdVectorOp).
           * With the current MLIR implementation, Operation::isBeforeInBlock() recomputes the *entire* operation order
           * of the block in case either lastVectorOp or createdVectorOp are new, which createdVectorOp would always be.
           * Depending on the size of the block and the number of created vector ops, this can waste a lot (!!) of time.
           */
          DenseMap<Superword*, Superword*> insertionPoints;

          /// Stores escaping users for each value.
          DenseMap<Value, SmallVector<Operation*, 1>> escapingUsers;

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
