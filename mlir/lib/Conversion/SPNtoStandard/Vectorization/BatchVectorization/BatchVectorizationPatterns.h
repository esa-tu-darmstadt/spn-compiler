//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_LIB_DIALECT_SPN_BATCHVECTORIZATION_BATCHVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_LIB_DIALECT_SPN_BATCHVECTORIZATION_BATCHVECTORIZATIONPATTERNS_H

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "SPN/SPNDialect.h"
#include "SPN/SPNOps.h"

namespace mlir {
  namespace spn {

    template<typename Op>
    class BatchVectorizationPattern : public mlir::OpRewritePattern<Op> {

    public:

      BatchVectorizationPattern(MLIRContext* context, unsigned _vectorWidth) : OpRewritePattern<Op>(context, 1),
                                                                               vectorWidth{_vectorWidth} {}

    protected:

      unsigned vectorWidth;

      mlir::VectorType createVectorType(Op op) const {
        return VectorType::get({vectorWidth}, op.getResult().getType());
      }

    };

    struct BatchVectorizeConstant : public BatchVectorizationPattern<ConstantOp> {

      using BatchVectorizationPattern<ConstantOp>::BatchVectorizationPattern;

      LogicalResult matchAndRewrite(ConstantOp op, PatternRewriter& rewriter) const override;

    };

    struct BatchVectorizeHistogram : public BatchVectorizationPattern<HistogramOp> {

      using BatchVectorizationPattern<HistogramOp>::BatchVectorizationPattern;

      LogicalResult matchAndRewrite(HistogramOp op, PatternRewriter& rewriter) const override;

    };

    struct BatchVectorizeCategorical : public BatchVectorizationPattern<CategoricalOp> {

      using BatchVectorizationPattern<CategoricalOp>::BatchVectorizationPattern;

      LogicalResult matchAndRewrite(CategoricalOp op, PatternRewriter& rewriter) const override;

    };

    struct BatchVectorizeGaussian : public BatchVectorizationPattern<GaussianOp> {

      using BatchVectorizationPattern<GaussianOp>::BatchVectorizationPattern;

      LogicalResult matchAndRewrite(GaussianOp op, PatternRewriter& rewriter) const override;

    };

    template<typename NAryOp>
    struct BatchVectorizeNAry : public BatchVectorizationPattern<NAryOp> {

      using BatchVectorizationPattern<NAryOp>::BatchVectorizationPattern;

      LogicalResult matchAndRewrite(NAryOp op, PatternRewriter& rewriter) const override {
        if (op.getResult().getType().template isa<VectorType>()) {
          return failure();
        }
        rewriter.template replaceOpWithNewOp<NAryOp>(op, this->createVectorType(op), op.operands());
        return success();
      }

    };

    using BatchVectorizeProduct = BatchVectorizeNAry<ProductOp>;
    using BatchVectorizeSum = BatchVectorizeNAry<SumOp>;

    static void populateBatchVectorizationPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                   unsigned vectorWidth) {
      patterns.insert<BatchVectorizeHistogram, BatchVectorizeCategorical, BatchVectorizeGaussian>(context, vectorWidth);
      patterns.insert<BatchVectorizeConstant>(context, vectorWidth);
      patterns.insert<BatchVectorizeProduct, BatchVectorizeSum>(context, vectorWidth);
    }

  }
}

#endif //SPNC_MLIR_LIB_DIALECT_SPN_BATCHVECTORIZATION_BATCHVECTORIZATIONPATTERNS_H
