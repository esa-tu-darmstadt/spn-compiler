//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_ANALYSIS_SLP_VECTOR_PATTERNS_H
#define SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_ANALYSIS_SLP_VECTOR_PATTERNS_H

#include <utility>

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/StandardTypes.h"
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

        bool assignedToVector(Operation* op) const {
          return vectorIndices.count(op);
        }

        std::map<Operation*, std::vector<size_t>> vectorIndices;
        std::vector<std::vector<Operation*>> const& vectors;
      };

      /// Pattern for transforming SPN Gaussian ops to a vectorized version.
      struct GaussianOpVectorization : public VectorizationPattern<GaussianOp> {

        using VectorizationPattern<GaussianOp>::VectorizationPattern;

        LogicalResult matchAndRewrite(GaussianOp op, PatternRewriter& rewriter) const override;

      };

      static void populateVectorizationPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                std::vector<std::vector<Operation*>> const& vectors) {
        patterns.insert<GaussianOpVectorization>(context, vectors);
      }
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_ANALYSIS_SLP_VECTOR_PATTERNS_H
