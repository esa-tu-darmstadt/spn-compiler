//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PATTERN_CANONICALIZATIONPATTERNS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PATTERN_CANONICALIZATIONPATTERNS_H

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "codegen/mlir/dialects/spn/SPNDialect.h"

namespace mlir {
  namespace spn {

    struct ReduceWeightedSumOp : public mlir::OpRewritePattern<WeightedSumOp> {

      explicit ReduceWeightedSumOp(MLIRContext* context) : OpRewritePattern<WeightedSumOp>(context, 1) {}

      PatternMatchResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;

    };

    struct ConstantFoldWeightedSumOp : public mlir::OpRewritePattern<WeightedSumOp> {

      explicit ConstantFoldWeightedSumOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      PatternMatchResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;

    };

    struct ReduceSumOp : public mlir::OpRewritePattern<SumOp> {

      explicit ReduceSumOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      PatternMatchResult matchAndRewrite(SumOp op, PatternRewriter& rewriter) const override;

    };

    struct ConstantFoldSumOp : public mlir::OpRewritePattern<SumOp> {

      explicit ConstantFoldSumOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      PatternMatchResult matchAndRewrite(SumOp op, PatternRewriter& rewriter) const override;

    };

    struct ReduceProductOp : public mlir::OpRewritePattern<ProductOp> {

      explicit ReduceProductOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      PatternMatchResult matchAndRewrite(ProductOp op, PatternRewriter& rewriter) const override;

    };

    struct ConstantFoldProductOp : public mlir::OpRewritePattern<ProductOp> {

      explicit ConstantFoldProductOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      PatternMatchResult matchAndRewrite(ProductOp op, PatternRewriter& rewriter) const override;

    };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PATTERN_CANONICALIZATIONPATTERNS_H
