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

    template<typename NAryOp>
    struct BinarizeNAryOp : public mlir::OpRewritePattern<NAryOp> {

      explicit BinarizeNAryOp(MLIRContext* context) : OpRewritePattern<NAryOp>(context, 1) {}

      PatternMatchResult matchAndRewrite(NAryOp op, PatternRewriter& rewriter) const override {
        if (op.getNumOperands() <= 2) {
          return BinarizeNAryOp<NAryOp>::matchFailure();
        }
        auto pivot = llvm::divideCeil(op.getNumOperands(), 2);
        SmallVector<Value, 10> leftAddends;
        SmallVector<Value, 10> rightAddends;
        int count = 0;
        for (auto a : op.operands()) {
          if (count < pivot) {
            leftAddends.push_back(a);
          } else {
            rightAddends.push_back(a);
          }
          ++count;
        }

        auto leftOp = rewriter.create<NAryOp>(op.getLoc(), leftAddends);
        auto rightOp = rewriter.create<NAryOp>(op.getLoc(), rightAddends);
        SmallVector<Value, 2> ops{leftOp, rightOp};
        auto newOp = rewriter.create<NAryOp>(op.getLoc(), ops);
        rewriter.replaceOp(op, {newOp});
        return BinarizeNAryOp<NAryOp>::matchSuccess();
      }

    };

    using BinarizeSumOp = BinarizeNAryOp<SumOp>;

    using BinarizeProductOp = BinarizeNAryOp<ProductOp>;

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PATTERN_SIMPLIFICATIONPATTERNS_H
