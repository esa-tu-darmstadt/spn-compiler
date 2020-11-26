//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_TYPE_ANALYSIS_TYPEPINNINGPATTERNS_H
#define SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_TYPE_ANALYSIS_TYPEPINNINGPATTERNS_H

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "SPN/SPNDialect.h"
#include "SPN/SPNOps.h"

namespace mlir {
  namespace spn {

    /// Abstract template for TypePinningPatterns.
    /// \tparam Op SPN dialect operation this pattern works on.
    template<typename Op>
    class TypePinningPattern : public mlir::OpRewritePattern<Op> {

    public:

      TypePinningPattern(MLIRContext* context, Type _newType) : OpRewritePattern<Op>(context, 1), newType{_newType} {}

    protected:

      Type newType;

    };

    ///
    /// Rewrite constant operation to use actual datatype instead of abstract
    /// SPN probability value type.
    struct TypePinConstant : public TypePinningPattern<ConstantOp> {
      using TypePinningPattern<ConstantOp>::TypePinningPattern;

      LogicalResult matchAndRewrite(ConstantOp op, PatternRewriter& rewriter) const override;
    };

    ///
    /// Rewrite histogram operation to use actual datatype instead of abstract
    /// SPN probability value type.
    struct TypePinHistogram : public TypePinningPattern<HistogramOp> {
      using TypePinningPattern<HistogramOp>::TypePinningPattern;

      LogicalResult matchAndRewrite(HistogramOp op, PatternRewriter& rewriter) const override;
    };

    ///
    /// Rewrite Categorical leaf to use actual datatype instead of
    /// abstract SPN probability value type.
    struct TypePinCategorical : public TypePinningPattern<CategoricalOp> {
      using TypePinningPattern<CategoricalOp>::TypePinningPattern;

      LogicalResult matchAndRewrite(CategoricalOp op, PatternRewriter& rewriter) const override;
    };

    ///
    /// Rewrite Gaussian leaf to use actual datatype instead of
    /// abstract SPN probability value type.
    struct TypePinGaussian : public TypePinningPattern<GaussianOp> {
      using TypePinningPattern<GaussianOp>::TypePinningPattern;

      LogicalResult matchAndRewrite(GaussianOp op, PatternRewriter& rewriter) const override;
    };

    ///
    /// Template to rewrite n-ary arithmetic operation to use actual datatype instead of abstract
    /// SPN probability value type.
    template<typename NAryOp>
    struct TypePinNAry : public TypePinningPattern<NAryOp> {
      using TypePinningPattern<NAryOp>::TypePinningPattern;

      LogicalResult matchAndRewrite(NAryOp op, PatternRewriter& rewriter) const override {
        if (!op.getResult().getType().template isa<ProbabilityType>()) {
          return failure();
        }
        rewriter.replaceOpWithNewOp<NAryOp>(op, this->newType, op.operands());
        return success();
      }
    };

    using TypePinProduct = TypePinNAry<ProductOp>;
    using TypePinSum = TypePinNAry<SumOp>;

    ///
    /// Rewrite weighted sum operation to use actual datatype instead of abstract
    /// SPN probability value type.
    struct TypePinWeightedSum : public TypePinningPattern<WeightedSumOp> {
      using TypePinningPattern<WeightedSumOp>::TypePinningPattern;

      LogicalResult matchAndRewrite(WeightedSumOp op, PatternRewriter& rewriter) const override;
    };

  }
}

#endif //SPNC_MLIR_DIALECTS_LIB_DIALECT_SPN_TYPE_ANALYSIS_TYPEPINNINGPATTERNS_H
