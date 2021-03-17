//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H

#include <utility>

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNDialect.h"

namespace mlir {
  namespace spn {
    namespace slp {

      template<typename Op>
      class VectorizationPattern : public mlir::OpRewritePattern<Op> {
      public:
        VectorizationPattern(MLIRContext* context, std::vector<std::vector<Operation*>> const& vectors)
            : OpRewritePattern<Op>(context, 1), vectors{vectors} {
          for (size_t i = 0; i < vectors.size(); ++i) {
            for (auto* op : vectors[i]) {
              vectorIndices[op].emplace_back(i);
            }
          }
        }
      protected:

        bool isAssignedToVector(Operation* op) const {
          return vectorIndices.count(op);
        }

        bool isVectorMixed(std::vector<Operation*> vector) const {
          return std::any_of(std::begin(vector), std::end(vector), [&](Operation* op) {
            return op->getName() != vector.front()->getName();
          });
        }

        std::map<Operation*, std::vector<size_t>> vectorIndices;
        std::vector<std::vector<Operation*>> const& vectors;
      };

      /// Pattern for transforming SPN sum ops to a vectorized version.
      struct SumOpVectorization : public VectorizationPattern<SumOp> {

        using VectorizationPattern<SumOp>::VectorizationPattern;

        LogicalResult matchAndRewrite(SumOp op, PatternRewriter& rewriter) const override;

      };

      static void populateVectorizationPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                std::vector<std::vector<Operation*>> const& vectors) {
        patterns.insert<SumOpVectorization>(context, vectors);
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
