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
#include "SLPVectorizationPatterns.h"
#include "LoSPN/LoSPNDialect.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class CostModel : public PatternVisitor {
        public:

          double getScalarCost(Value const& value) {
            auto const& entry = cachedScalarCosts.try_emplace(value, 0);
            if (entry.second) {
              entry.first->second = computeScalarCost(value);
            }
            return entry.first->second;
          }

          double getScalarTreeCost(Value const& root) {
            llvm_unreachable("TODO: scalar visitor cost model for LoSPN ops");
          }

          double getSuperwordCost(Superword* superword, SLPVectorizationPattern* pattern) {
            auto it = cachedSuperwordCosts.try_emplace(superword, 0);
            if (it.second) {
              pattern->accept(*this, superword);
              it.first->second = cost;
            }
            return it.first->second;
          }

        protected:
          virtual double computeScalarCost(Value const& value) = 0;
          virtual double singleElementExtractionCost() = 0;
          double cost;
        private:
          DenseMap<Value, double> cachedScalarCosts;
          DenseMap<Value, double> cachedScalarTreeCosts;
          DenseMap<Superword*, double> cachedSuperwordCosts;
          DenseMap<Superword*, double> cachedSuperwordTreeCosts;
        };

        class UnitCostModel : public CostModel {
          double computeScalarCost(Value const& value) override;
          double singleElementExtractionCost() override;
          void visit(BroadcastSuperword* pattern, Superword* superword) override;
          void visit(BroadcastInsertSuperword* pattern, Superword* superword) override;
          void visit(VectorizeConstant* pattern, Superword* superword) override;
          void visit(VectorizeBatchRead* pattern, Superword* superword) override;
          void visit(VectorizeAdd* pattern, Superword* superword) override;
          void visit(VectorizeMul* pattern, Superword* superword) override;
          void visit(VectorizeGaussian* pattern, Superword* superword) override;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_COSTMODEL_H
