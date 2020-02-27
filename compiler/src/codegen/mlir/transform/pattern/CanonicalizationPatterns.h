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

    template<typename NAryOp>
    struct ReduceNAryOp : public mlir::OpRewritePattern<NAryOp> {

      explicit ReduceNAryOp(MLIRContext* context) : OpRewritePattern<NAryOp>(context, 1) {}

      PatternMatchResult matchAndRewrite(NAryOp op, PatternRewriter& rewriter) const override {
        if (op.getNumOperands() > 1) {
          return ReduceNAryOp<NAryOp>::matchFailure();
        }

        if (op.getNumOperands() == 0) {
          rewriter.eraseOp(op.getOperation());
          return ReduceNAryOp<NAryOp>::matchSuccess();
        }
        assert(op.getNumOperands() == 1 && "Expecting only a single operand!");
        rewriter.replaceOp(op, {op.operands()[0]});
        return ReduceNAryOp<NAryOp>::matchSuccess();
      }

    };

    template<typename NAryOp, typename Acc, int Initial>
    struct ConstantFoldNAryOp : public mlir::OpRewritePattern<NAryOp> {

      explicit ConstantFoldNAryOp(MLIRContext* context) : OpRewritePattern<NAryOp>(context, 1) {}

      PatternMatchResult matchAndRewrite(NAryOp op, PatternRewriter& rewriter) const override {
        SmallVector<Value, 10> operands;
        auto acc = (double) Initial;
        size_t constantOps = 0;
        Acc accOp;
        for (auto m : op.operands()) {
          if (auto constantOp = dyn_cast_or_null<ConstantOp>(m.getDefiningOp())) {
            acc = accOp(acc, constantOp.value().convertToDouble());
            ++constantOps;
          } else {
            operands.push_back(m);
          }
        }
        if (constantOps <= 1) {
          return ConstantFoldNAryOp<NAryOp, Acc, Initial>::matchFailure();
        }

        if (acc != ((double) Initial)) {
          operands.push_back(rewriter.create<ConstantOp>(op.getLoc(), acc));
        }
        auto newOp = rewriter.create<NAryOp>(op.getLoc(), operands);
        rewriter.replaceOp(op, {newOp});
        return ConstantFoldNAryOp<NAryOp, Acc, Initial>::matchSuccess();
      }
    };

    using ReduceSumOp = ReduceNAryOp<SumOp>;
    using ReduceProductOp = ReduceNAryOp<ProductOp>;

    using ConstantFoldSumOp = ConstantFoldNAryOp<SumOp, std::plus<>, 0>;
    using ConstantFoldProductOp = ConstantFoldNAryOp<ProductOp, std::multiplies<>, 1>;

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_TRANSFORM_PATTERN_CANONICALIZATIONPATTERNS_H
