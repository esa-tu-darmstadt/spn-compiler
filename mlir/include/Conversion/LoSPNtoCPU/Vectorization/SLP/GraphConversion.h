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
          ValuePosition getWordContainingValue(Value const& value) const;
          // Callback registration.
          void addVectorCallback(std::function<void(Superword*)> callback);
          void addScalarCallback(std::function<void(Value)> callback);
          void addExtractionCallback(std::function<void(Value)> callback);
        private:
          SmallPtrSet<Superword*, 32> computedSuperwords;
          SmallPtrSet<Value, 32> computedScalarValues;
          DenseMap<Value, ValuePosition> extractableScalarValues;
          // Callbacks for when a superword was converted.
          SmallVector<std::function<void(Superword*)>> vectorCallbacks;
          // Callbacks for when a scalar value is used as input for some vector.
          SmallVector<std::function<void(Value)>> scalarCallbacks;
          // Callbacks for when an extraction for some value has been created.
          SmallVector<std::function<void(Value)>> extractionCallbacks;
        };

        class ConversionPlan {
        public:
          struct Step {
            Step(Superword* superword, SLPVectorizationPattern* pattern) : superword{superword}, pattern{pattern} {}
            Superword* superword;
            SLPVectorizationPattern* pattern;
          };
          ConversionPlan(std::shared_ptr<ConversionState> conversionState);
          void addConversionStep(Superword* superword, SLPVectorizationPattern* pattern);
        private:
          SmallVector<Step, 32> plan;
          std::shared_ptr<ConversionState> conversionState;
          ScalarValueVisitor scalarVisitor;
          std::shared_ptr<CostModel> costModel;
        };

        enum ElementFlag {
          /// Erase all elements after conversion.
          KeepNone,
          /// Erase all but the first element after conversion.
          KeepFirst,
          /// Do not erase any element after conversion.
          KeepAll
        };

        class ConversionManager {

        public:

          ConversionManager(PatternRewriter& rewriter, std::shared_ptr<ConversionState> conversionState);

          void initConversion(Superword* root, Block* block);
          void finishConversion(Block* block);

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
          std::shared_ptr<ConversionState> conversionState;

          Value latestCreation;

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
