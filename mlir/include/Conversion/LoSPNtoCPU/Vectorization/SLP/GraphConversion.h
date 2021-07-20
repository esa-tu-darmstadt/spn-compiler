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
          void startConversion(std::shared_ptr<Superword> root);
          void finishConversion();
          void cancelConversion();

          bool alreadyComputed(Superword* superword) const;
          bool alreadyComputed(Value value) const;
          bool isExtractable(Value value);

          void markComputed(Value value);
          void markComputed(Superword* superword, Value value);
          void markExtracted(Value value);

          Value getValue(Superword* superword) const;
          ValuePosition getSuperwordContainingValue(Value value) const;

          SmallVector<Superword*> unconvertedPostOrder() const;

          // Callback registration.
          /// Callbacks for when a superword was converted and for when its value has been removed because the graph
          /// it was contained in was not deemed profitable.
          void addVectorCallbacks(std::function<void(Superword*)> createCallback,
                                  std::function<void(Superword*)> undoCallback);
          /// Callbacks for when a scalar value is being used as input for some vector and for when a scalar that was
          /// previously used as input for some vector is no longer an input because the corresponding graph was
          /// not deemed profitable.
          void addScalarCallbacks(std::function<void(Value)> inputCallback, std::function<void(Value)> undoCallback);
          /// Callbacks for when an extraction for some value has been created and for when an extraction for some
          /// value has been undone because the corresponding graph was not deemed profitable.
          void addExtractionCallbacks(std::function<void(Value)> extractCallback,
                                      std::function<void(Value)> undoCallback);
        private:
          // Take ownership of the graphs to prevent dangling pointers when they go out of scope.
          SmallVector<std::shared_ptr<Superword>, 5> correspondingGraphs;

          // For bookkeeping of computed superwords and values.
          struct ConversionData {

            bool alreadyComputed(Value value) const {
              return computedScalarValues.contains(value) || extractedScalarValues.contains(value);
            }

            void clear() {
              computedScalarValues.clear();
              computedSuperwords.clear();
              extractedScalarValues.clear();
              extractableScalarValues.clear();
            }

            void mergeWith(ConversionData& other) {
              computedScalarValues.insert(std::begin(other.computedScalarValues), std::end(other.computedScalarValues));
              computedSuperwords.copyFrom(other.computedSuperwords);
              extractedScalarValues.insert(std::begin(other.extractedScalarValues),
                                           std::end(other.extractedScalarValues));
              extractableScalarValues.copyFrom(other.extractableScalarValues);
            }
            /// Scalar values that are marked as computed (e.g. because they're used as inputs for vectors).
            SmallPtrSet<Value, 32> computedScalarValues;
            /// Superwords that are marked as computed.
            DenseMap<Superword*, Value> computedSuperwords;
            /// Extractions that have taken place.
            SmallPtrSet<Value, 32> extractedScalarValues;
            /// Store vector element data for faster extraction lookup.
            DenseMap<Value, ValuePosition> extractableScalarValues;
          };

          // Permanent conversion data.
          ConversionData permanentData;
          // Temporary conversion data. These might need to be 'undone' when a graph is not deemed profitable.
          ConversionData temporaryData;
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

          ConversionManager(RewriterBase& rewriter, Block* block, std::shared_ptr<CostModel> costModel);

          SmallVector<Superword*> startConversion(SLPGraph const& graph);
          void finishConversion();
          void cancelConversion();

          void setupConversionFor(Superword* superword, SLPVectorizationPattern const* pattern);
          void update(Superword* superword, Value operation, SLPVectorizationPattern const* appliedPattern);

          Value getValue(Superword* superword) const;
          Value getOrCreateConstant(Location const& loc, Attribute const& attribute);
          ConversionState& getConversionState() const;

        private:
          bool hasEscapingUsers(Value value) const;
          Value getOrExtractValue(Value value);
          void reorderOperations();

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

          /// For creating constants, setting insertion points, creating extractions, ....
          RewriterBase& rewriter;

          /// Helps avoid creating duplicate constants.
          OperationFolder folder;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
