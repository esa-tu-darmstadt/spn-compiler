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
          explicit operator bool() const {
            return superword;
          }
          Superword* superword;
          size_t index;
        };

        class ConversionState {
        public:
          ConversionState() = default;
          ConversionState(std::shared_ptr<Superword> root, std::shared_ptr<ConversionState> parentState);

          bool alreadyComputed(Superword* superword) const;
          bool alreadyComputed(Value value) const;
          bool isExtractable(Value value);
          bool isDead(Operation* op) const;

          void markComputed(Value value);
          void markComputed(Superword* superword, Value value);
          void markExtracted(Value value);
          void markDead(Operation* op);

          Value getValue(Superword* superword) const;
          ValuePosition getSuperwordContainingValue(Value value) const;

          SmallPtrSet<Operation*, 32> getDeadOps() const;
          SmallVector<Superword*> unconvertedPostOrder() const;
          void undoChanges() const;

          std::shared_ptr<ConversionState> getParentState() const;

          // Callback registration.
          /// Callback for when a superword was converted.
          void addVectorCallback(std::function<void(Superword*)> callback);
          /// Callback for when a scalar value is being used as input for some vector.
          void addScalarCallback(std::function<void(Value)> callback);
          /// Callback for when an extraction for some value has been created.
          void addExtractionCallback(std::function<void(Value)> callback);
          /// Callback for when the conversion of a superword has been undone because its graph was not deemed
          /// profitable.
          void addVectorUndoCallback(std::function<void(Superword*)> callback);
          /// Callback for when a scalar that was previously used as input for some vector is no longer an input
          /// because the corresponding graph was not deemed profitable.
          void addScalarUndoCallback(std::function<void(Value)> callback);
          /// Callback for when an extraction for some value has been undone because the corresponding graph was
          /// not deemed profitable.
          void addExtractionUndoCallback(std::function<void(Value)> callback);
        private:
          // Take ownership of the graph to prevent dangling pointers when it goes out of scope.
          std::shared_ptr<Superword> correspondingGraph;
          /// Enables recursive lookups from previous graph conversions.
          std::shared_ptr<ConversionState> parentState;
          /// Actual conversion data.
          SmallPtrSet<Value, 32> computedScalarValues;
          DenseMap<Superword*, Value> computedSuperwords;
          SmallPtrSet<Value, 32> extractedScalarValues;
          /// Store vector element data for faster extraction lookup.
          DenseMap<Value, ValuePosition> extractableScalarValues;
          /// Store dead operations to prevent erasing them more than once.
          SmallPtrSet<Operation*, 32> deadOps;
          // Callback data.
          SmallVector<std::function<void(Superword*)>> vectorCallbacks;
          SmallVector<std::function<void(Superword*)>> vectorUndoCallbacks;
          SmallVector<std::function<void(Value)>> scalarCallbacks;
          SmallVector<std::function<void(Value)>> scalarUndoCallbacks;
          SmallVector<std::function<void(Value)>> extractionCallbacks;
          SmallVector<std::function<void(Value)>> extractionUndoCallbacks;
        };

        class ConversionManager {

        public:

          ConversionManager(PatternRewriter& rewriter, Block* block, std::shared_ptr<CostModel> costModel);

          SmallVector<Superword*> initConversion(SLPGraph const& graph);
          void finishConversion();
          void undoConversion();

          void setupConversionFor(Superword* superword, SLPVectorizationPattern const* pattern);
          void update(Superword* superword, Value operation, SLPVectorizationPattern const* appliedPattern);

          Value getValue(Superword* superword) const;
          Value getOrCreateConstant(Location const& loc, Attribute const& attribute);
          void eraseAllDeadOps();

        private:
          bool hasEscapingUsers(Value value) const;
          Value getOrExtractValue(Value value);
          void updateOperand(Operation* op, Value oldOperand, Value newOperand);

          Block* block;
          std::shared_ptr<CostModel> costModel;
          std::shared_ptr<ConversionState> conversionState;

          /// Stores escaping users for each value.
          DenseMap<Value, SmallVector<Operation*, 1>> escapingUsers;

          /// For reverting back to pre-conversion block states.
          SmallVector<Operation*, 32> originalOperations;
          DenseMap<Operation*, SmallVector<Value, 2>> originalOperands;

          /// Helps find out which vector elements can be erased.
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
