//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PATTERN_SIMPLIFICATIONPATTERNS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PATTERN_SIMPLIFICATIONPATTERNS_H

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "codegen/mlir/dialects/spn/SPNDialect.h"

namespace mlir {
  namespace spn {

    struct BinarizeWeightedSumOp : public mlir::OpRewritePattern<WeightedSumOp> {

      explicit BinarizeWeightedSumOp(MLIRContext* context)
          : OpRewritePattern<WeightedSumOp>(context, 1) {}

      PatternMatchResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;

    };

    struct SplitWeightedSumOp : public mlir::OpRewritePattern<WeightedSumOp> {

      explicit SplitWeightedSumOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      PatternMatchResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;

    };

    struct BinarizeSumOp : public mlir::OpRewritePattern<SumOp> {

      explicit BinarizeSumOp(MLIRContext* context) : OpRewritePattern<SumOp>(context, 1) {}

      PatternMatchResult matchAndRewrite(SumOp op, PatternRewriter& rewriter) const override;

    };

    struct BinarizeProductOp : public mlir::OpRewritePattern<ProductOp> {

      explicit BinarizeProductOp(MLIRContext* context) : OpRewritePattern<ProductOp>(context, 1) {}

      PatternMatchResult matchAndRewrite(ProductOp op, PatternRewriter& rewriter) const override;

    };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PATTERN_SIMPLIFICATIONPATTERNS_H
