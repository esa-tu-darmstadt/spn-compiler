//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_ANALYSIS_SLP_VECTOR_PATTERNS_H
#define SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_ANALYSIS_SLP_VECTOR_PATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNDialect.h"

namespace mlir {
  namespace spn {
    namespace slp {

      /// Pattern for transforming SPN Gaussian ops to a vectorized version.
      struct GaussianOpVectorization : public OpRewritePattern<GaussianOp> {

        using OpRewritePattern<GaussianOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(GaussianOp op, PatternRewriter& rewriter) const override;

      };

      /// Populate list with all patterns required to transform SPN operations into vectorized operations.
      /// \param patterns Pattern list to fill.
      /// \param context MLIR context.
      /// \param typeConverter Type converter.
      static void populateVectorizationPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                TypeConverter& typeConverter) {
        patterns.insert<GaussianOpVectorization>(context);
      }
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_ANALYSIS_SLP_VECTOR_PATTERNS_H
