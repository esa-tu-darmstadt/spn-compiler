//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H

#include "SLPGraph.h"
#include "GraphConversion.h"
#include "PatternVisitors.h"
#include "SLPPatternMatch.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        /// The cost model calculates the profitability of the vectorization. It is also used to determine the most
        /// profitable pattern for each superword, depending on the current vectorization state. It extends the
        /// PatternVisitor class to make pattern cost computations easier.
        class CostModel : public PatternVisitor {
        public:
          /// The base constructor for every cost model. Requires a pattern applicator because the cost model needs to
          /// know which patterns are applicable.
          explicit CostModel(SLPPatternApplicator const& applicator);
          /// Computes the cost of a scalar value recursively, i.e.
          /// \code cost(value) = cost(value op) + sum(cost(operand i)) \endcode for each operand \em i.
          /// Depending on the vectorization state, cost may be zero if the value has already been marked as computed.
          double getScalarCost(Value value);
          /// Computes the cost of a superword if the provided pattern were to be applied. The computation is done in
          /// a similar fashion to CostModel::getScalarCost(Value) i.e.
          /// \code cost(superword, pattern) = cost(pattern) + sum(cost(superword i, superword i pattern)) \endcode
          /// for each operand superword i.
          double getSuperwordCost(Superword* superword, SLPVectorizationPattern* pattern);
          /// Determines if an extraction for the provided value would be more profitable than computing it in a scalar
          /// fashion.
          bool isExtractionProfitable(Value value);
          /// Set the conversion state which keeps track of values that have been marked as computed. This is required
          /// for accurate, conversion state-aware cost computations.
          void setConversionState(std::shared_ptr<ConversionState> newConversionState);
          /// Computes the cost of an entire block. The set of dead ops is necessary to only consider operations that
          /// are actually relevant.
          double getBlockCost(Block* block, SmallPtrSetImpl<Operation*> const& deadOps) const;
        protected:
          virtual double computeScalarCost(Value value) const = 0;
          virtual double computeExtractionCost(Superword* superword, size_t index) const = 0;
          double cost;
          /// For computation of required scalar values.
          LeafPatternVisitor leafVisitor;
          std::shared_ptr<ConversionState> conversionState;
        private:
          void updateCost(Value value, double newCost, bool updateUses);
          double getExtractionCost(Value value) const;
          static constexpr double MAX_COST = std::numeric_limits<double>::max();
          DenseMap<Value, double> cachedScalarCost;
          SLPPatternApplicator const& patternApplicator;
        };

        /// The unit cost model assumes cost of 1 for every MLIR operation. This means, that if a pattern lowers some
        /// superword to a group of 5 statements in whatever dialect, the pattern's cost will be 5. It does not take
        /// into account how many LLVM operations or even machine code instructions these 5 will then be lowered to.
        class UnitCostModel : public CostModel {
          using CostModel::CostModel;
          double computeScalarCost(Value value) const override;
          double computeExtractionCost(Superword* superword, size_t index) const override;
          void visitDefault(SLPVectorizationPattern const* pattern, Superword const* superword) override;
          void visit(BroadcastSuperword const* pattern, Superword const* superword) override;
          void visit(BroadcastInsertSuperword const* pattern, Superword const* superword) override;
          void visit(VectorizeConstant const* pattern, Superword const* superword) override;
          void visit(VectorizeSPNConstant const* pattern, Superword const* superword) override;
          void visit(VectorizeGaussian const* pattern, Superword const* superword) override;
          void visit(VectorizeLogAdd const* pattern, Superword const* superword) override;
          void visit(VectorizeLogGaussian const* pattern, Superword const* superword) override;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H
