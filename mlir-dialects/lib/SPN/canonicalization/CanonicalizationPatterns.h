//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_LIB_SPN_CANONICALIZATION_CANONICALIZATIONPATTERNS_H
#define SPNC_MLIR_DIALECTS_LIB_SPN_CANONICALIZATION_CANONICALIZATIONPATTERNS_H

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "SPN/SPNDialect.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNAttributes.h"

namespace mlir {
  namespace spn {
    ///
    /// OpRewritePattern to reduce weighted sums as far as possible.
    /// Operations with no operand are deleted.
    /// For operations with a single operand, all uses are replaced by the operand.
    /// If all weights are 1.0, the operation is replaced by an ordinary sum.
    struct ReduceWeightedSumOp : public mlir::OpRewritePattern<WeightedSumOp> {

      /// Constructor.
      /// \param context Surrounding MLIRContext.
      explicit ReduceWeightedSumOp(MLIRContext* context) : OpRewritePattern<WeightedSumOp>(context, 1) {}

      LogicalResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;

    };

    ///
    /// OpRewritePattern merging constant operands of a weighted sum into a single child node.
    struct ConstantFoldWeightedSumOp : public mlir::OpRewritePattern<WeightedSumOp> {

      /// Constructor.
      /// \param context Surrounding MLIRContext.
      explicit ConstantFoldWeightedSumOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      LogicalResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;

    };

    /// OpRewritePattern to reduce n-ary operations as far as possible.
    /// For operations with a single operand, all uses are replaced by the operand.
    /// Can only be applied to operations from the SPN dialects inheriting from SPN_NAry_Op.
    /// \tparam NAryOp Operation type.
    template<typename NAryOp>
    struct ReduceNAryOp : public mlir::OpRewritePattern<NAryOp> {

      /// Constructor.
      /// \param context Surrounding MLIRContext.
      explicit ReduceNAryOp(MLIRContext* context) : OpRewritePattern<NAryOp>(context, 1) {}

      LogicalResult matchAndRewrite(NAryOp op, PatternRewriter& rewriter) const override {
        if (op.getNumOperands() > 1) {
          return failure();
        }
        if (op.getNumOperands() == 0) {
          rewriter.replaceOpWithNewOp<ConstantOp>(op, 0);
          return success();
        }

        assert(op.getNumOperands() == 1 && "Expecting only a single operand!");
        rewriter.replaceOp(op, {op.operands()[0]});
        return success();
      }

    };

    using ReduceSumOp = ReduceNAryOp<SumOp>;
    using ReduceProductOp = ReduceNAryOp<ProductOp>;

    ///
    /// OpRewritePattern to merge constant operands of n-ary (non-weighted) sums.
    struct ConstantFoldSumOp : public mlir::OpRewritePattern<SumOp> {

      /// Constructor.
      /// \param context Surrounding MLIRContext.
      explicit ConstantFoldSumOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      LogicalResult matchAndRewrite(SumOp op, PatternRewriter& rewriter) const override;

    };

    ///
    /// OpRewritePattern to merge constant operands of n-ary products.
    struct ConstantFoldProductOp : public mlir::OpRewritePattern<ProductOp> {

      /// Constructor.
      /// \param context Surrounding MLIRContext.
      explicit ConstantFoldProductOp(MLIRContext* context) : OpRewritePattern(context, 1) {}

      LogicalResult matchAndRewrite(ProductOp op, PatternRewriter& rewriter) const override;

    };

    namespace detail {

      /// Perform constant folding for n-ary operation.
      /// \tparam NAryOp Operation type.
      /// \tparam accFunc Function to use to for merging constant operands.
      /// \param op Operation to perform operand constant folding on.
      /// \param operands List of non-constant operands, will be filled by this function.
      /// \param initialValue Initial (neutral) value of the constant accumulator.
      /// \return NaN if no constant operands where found, the merged constant value otherwise.
      template<typename NAryOp, typename accFunc>
      double constantFoldOperands(NAryOp op, SmallVectorImpl<Value>& operands, double initialValue) {
        auto acc = initialValue;
        size_t constantOps = 0;
        accFunc accOp;
        for (auto m : op.operands()) {
          if (auto constantOp = dyn_cast_or_null<ConstantOp>(m.getDefiningOp())) {
            acc = accOp(acc, constantOp.value().convertToDouble());
            ++constantOps;
          } else {
            operands.push_back(m);
          }
        }
        if (constantOps == 0) {
          return std::nan("");
        }
        return acc;
      }

    } // end of namespace detail

  } // end of namespace spn
} // end of namespace mlir

#endif //SPNC_MLIR_DIALECTS_LIB_SPN_CANONICALIZATION_CANONICALIZATIONPATTERNS_H
