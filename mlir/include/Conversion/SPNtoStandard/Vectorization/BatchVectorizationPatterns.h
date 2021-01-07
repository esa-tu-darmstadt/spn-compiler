//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_LIB_DIALECT_SPN_BATCHVECTORIZATION_BATCHVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_LIB_DIALECT_SPN_BATCHVECTORIZATION_BATCHVECTORIZATIONPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNDialect.h"

namespace mlir {
  namespace spn {

    struct BatchVectorizeJointLowering : public OpConversionPattern<JointQuery> {

      using OpConversionPattern<JointQuery>::OpConversionPattern;

      LogicalResult matchAndRewrite(JointQuery op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    struct BatchVectorizeGaussian : public OpConversionPattern<GaussianOp> {

      using OpConversionPattern<GaussianOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(GaussianOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    /// Populate list with all patterns required to batch vectorize SPN dialect operations, lowering
    /// to a combination of Standard and Vector dialect.
    /// \param patterns Pattern list to fill.
    /// \param context MLIR context.
    /// \param typeConverter Type converter.
    static void populateSPNBatchVectorizePatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                  TypeConverter& typeConverter) {
      patterns.insert<BatchVectorizeGaussian>(typeConverter, context, 2);
      patterns.insert<BatchVectorizeJointLowering>(typeConverter, context, 5);
    }

  }
}

#endif //SPNC_MLIR_LIB_DIALECT_SPN_BATCHVECTORIZATION_BATCHVECTORIZATIONPATTERNS_H
