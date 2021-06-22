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
#include "PatternVisitors.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class CostModel;

        struct ValuePosition {
          ValuePosition() : superword{nullptr}, index{0} {}
          ValuePosition(Superword* superword, size_t index) : superword{superword}, index{index} {}
          Superword* superword;
          size_t index;
        };

        class ConversionState {
        public:
          bool alreadyComputed(Superword* superword) const;
          bool alreadyComputed(Value const& value) const;
          void markComputed(Superword* superword);
          void markComputed(Value const& value);
          void markExtracted(Value const& value);
          ValuePosition getWordContainingValue(Value const& value) const;
          // Callback registration.
          void addVectorCallback(std::function<void(Superword*)> callback);
          void addScalarCallback(std::function<void(Value)> callback);
          void addExtractionCallback(std::function<void(Value)> callback);
        private:
          SmallPtrSet<Superword*, 32> computedSuperwords;
          SmallPtrSet<Value, 32> computedScalarValues;
          DenseMap<Value, ValuePosition> extractableScalarValues;
          /// Callbacks for when a superword was converted.
          SmallVector<std::function<void(Superword*)>> vectorCallbacks;
          /// Callbacks for when a scalar value is used as input for some vector.
          SmallVector<std::function<void(Value)>> scalarCallbacks;
          /// Callbacks for when an extraction for some value has been created.
          SmallVector<std::function<void(Value)>> extractionCallbacks;
        };

        class ConversionManager {

        public:

          ConversionManager(PatternRewriter& rewriter,
                            std::shared_ptr<ConversionState> conversionState,
                            std::shared_ptr<CostModel> costModel);

          void initConversion(Superword* root, Block* block);
          void finishConversion(Block* block);
          ArrayRef<Superword*> conversionOrder() const;

          void setupConversionFor(Superword* superword, SLPVectorizationPattern const* pattern);
          void update(Superword* superword, Value const& operation, SLPVectorizationPattern const* appliedPattern);

          Value getValue(Superword* superword) const;
          Value getOrCreateConstant(Location const& loc, Attribute const& attribute);

        private:
          bool wasConverted(Superword* superword) const;
          bool hasEscapingUsers(Value const& value) const;
          Value getOrExtractValue(Value const& value);

          SmallVector<Superword*> order;

          DenseMap<Superword*, Value> vectorOperations;
          std::shared_ptr<ConversionState> conversionState;

          Value latestCreation;
          std::shared_ptr<CostModel> costModel;

          /// Stores escaping users for each value.
          DenseMap<Value, SmallVector<Operation*, 1>> escapingUsers;

          /// For finding out which vector elements can be erased.
          LeafPatternVisitor leafVisitor;

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
