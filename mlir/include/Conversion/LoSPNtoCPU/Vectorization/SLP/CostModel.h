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

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class CostModel : public PatternVisitor {
        public:
          double getScalarCost(Value const& value);
          double getSuperwordCost(Superword* superword, SLPVectorizationPattern* pattern);
          bool isExtractionProfitable(Value const& value);
          void setConversionState(std::shared_ptr<ConversionState> const& newConversionState);
        protected:
          virtual double computeScalarCost(Value const& value) = 0;
          virtual double computeExtractionCost(Superword* superword, size_t index) = 0;
          double cost;
          // For insertion/extraction cost computation.
          ScalarValueVisitor scalarVisitor;
          std::shared_ptr<ConversionState> conversionState;
        private:
          void updateCost(Value const& value, double newCost, bool updateUses);
          double getExtractionCost(Value const& value);
          static constexpr double MAX_COST = std::numeric_limits<double>::max();
          DenseMap<Value, double> cachedScalarCost;
        };

        class UnitCostModel : public CostModel {
          double computeScalarCost(Value const& value) override;
          double computeExtractionCost(Superword* superword, size_t index) override;
          void visitDefault(SLPVectorizationPattern* pattern, Superword* superword) override;
          void visit(BroadcastSuperword* pattern, Superword* superword) override;
          void visit(BroadcastInsertSuperword* pattern, Superword* superword) override;
          void visit(VectorizeConstant* pattern, Superword* superword) override;
          void visit(VectorizeGaussian* pattern, Superword* superword) override;
          void visit(VectorizeLogConstant* pattern, Superword* superword) override;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H
