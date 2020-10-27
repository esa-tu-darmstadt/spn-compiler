//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_LIB_SPN_SIMPLIFICATION_SIMPLIFICATIONPATTERNS_H
#define SPNC_MLIR_DIALECTS_LIB_SPN_SIMPLIFICATION_SIMPLIFICATIONPATTERNS_H

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "SPN/SPNDialect.h"
#include "SPN/SPNOps.h"

namespace mlir {
  namespace spn {

    ///
    /// OpRewritePattern turning an n-ary weighted sum into a tree of
    /// two-input operations.
    struct BinarizeWeightedSumOp : public mlir::OpRewritePattern<WeightedSumOp> {

      /// Constructor.
      /// \param context Surrounding MLIRContext.
      explicit BinarizeWeightedSumOp(MLIRContext* context)
          : OpRewritePattern<WeightedSumOp>(context, 1) {}

      LogicalResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;

    };

    ///
    /// OpRewritePattern decomposing a weighted sum into simple sum and multiplications.
    struct SplitWeightedSumOp : public mlir::OpRewritePattern<WeightedSumOp> {

      /// Constructor.
      /// \param context Surrounding MLIRContext.
      explicit SplitWeightedSumOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      LogicalResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;

    };

    /// Templated implementation of an OpRewritePattern decomposing an n-ary arithmetic
    /// operation into a tree of two-input operations. Can only be applied to operations
    /// from the SPN dialects inheriting from SPN_NAry_Op.
    /// \tparam NAryOp Operation type.
    template<typename NAryOp>
    struct BinarizeNAryOp : public mlir::OpRewritePattern<NAryOp> {

      /// Constructor.
      /// \param context Surrounding MLIRContext.
      explicit BinarizeNAryOp(MLIRContext* context) : OpRewritePattern<NAryOp>(context, 1) {}

      LogicalResult matchAndRewrite(NAryOp op, PatternRewriter& rewriter) const override {
        if (op.getNumOperands() <= 2) {
          return failure();
        }
        auto pivot = llvm::divideCeil(op.getNumOperands(), 2);
        SmallVector<Value, 10> leftOperands;
        SmallVector<Value, 10> rightOperands;
        unsigned count = 0;
        // Equally split the operands into two halves and recurse,
        // yielding a balanced tree.
        for (auto a : op.operands()) {
          if (count < pivot) {
            leftOperands.push_back(a);
          } else {
            rightOperands.push_back(a);
          }
          ++count;
        }

        auto leftOp = rewriter.create<NAryOp>(op.getLoc(), leftOperands);
        auto rightOp = rewriter.create<NAryOp>(op.getLoc(), rightOperands);
        SmallVector<Value, 2> ops{leftOp, rightOp};
        rewriter.replaceOpWithNewOp<NAryOp>(op, ops);
        return success();
      }

    };

    using BinarizeSumOp = BinarizeNAryOp<SumOp>;

    using BinarizeProductOp = BinarizeNAryOp<ProductOp>;

  }
}

#endif //SPNC_MLIR_DIALECTS_LIB_SPN_SIMPLIFICATION_SIMPLIFICATIONPATTERNS_H
