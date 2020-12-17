//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDPATTERNS_H
#define SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNDialect.h"

namespace mlir {
  namespace spn {

    ///
    /// Pattern for lowering SPN constant op to Standard dialect.
    struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {

      using OpConversionPattern<ConstantOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    ///
    /// Pattern for lowering SPN return op to Standard dialect.
    struct ReturnOpLowering : public OpConversionPattern<ReturnOp> {

      using OpConversionPattern<ReturnOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(ReturnOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    ///
    /// Pattern for lowering SPN Gaussian leaf to actual computation of
    /// Gaussian distribution in the Standard dialect.
    struct GaussionOpLowering : public OpConversionPattern<GaussianOp> {

      using OpConversionPattern<GaussianOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(GaussianOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    ///
    /// Pattern for lowering SPN joint query operation with batch size 1 to Standard dialect.
    struct SingleJointLowering : public OpConversionPattern<JointQuery> {

      using OpConversionPattern<JointQuery>::OpConversionPattern;

      LogicalResult matchAndRewrite(JointQuery op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    ///
    /// Pattern for lowering SPN joint query operation with batch size >1 to a combination
    /// of Standard and SCF (structured control flow) dialect.
    struct BatchJointLowering : public OpConversionPattern<JointQuery> {

      using OpConversionPattern<JointQuery>::OpConversionPattern;

      LogicalResult matchAndRewrite(JointQuery op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    struct BatchVectorizeJointLowering : public OpConversionPattern<JointQuery> {

      using OpConversionPattern<JointQuery>::OpConversionPattern;

      LogicalResult matchAndRewrite(JointQuery op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    /// Template for patterns lowering SPN n-ary arithmetic operations to Standard dialect.
    /// Will only work if the arithmetic is actually happening on floating-point data types.
    /// \tparam SourceOp SPN dialect operation to lower.
    /// \tparam TargetOp Standard dialect operation to lower to.
    template<typename SourceOp, typename TargetOp>
    class FloatArithmeticOpLowering : public OpConversionPattern<SourceOp> {

      using OpConversionPattern<SourceOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override {
        auto opType = operands[0].getType();
        if (!opType.isIntOrFloat() || opType.isIntOrIndex()) {
          // Translate only arithmetic operations operating on floating-point data types.
          return failure();
        }

        if (op.getNumOperands() > 2 || operands.size() != op.getNumOperands()) {
          return failure();
        }

        rewriter.replaceOpWithNewOp<TargetOp>(op, operands[0], operands[1]);
        return success();
      }
    };

    using FloatProductLowering = FloatArithmeticOpLowering<ProductOp, mlir::MulFOp>;
    using FLoatSumLowering = FloatArithmeticOpLowering<SumOp, mlir::AddFOp>;

    /// Populate list with all patterns required to lower SPN dialect operations to Standard dialect.
    /// \param patterns Pattern list to fill.
    /// \param context MLIR context.
    /// \param typeConverter Type converter.
    static void populateSPNtoStandardConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                        TypeConverter& typeConverter) {
      patterns.insert<ReturnOpLowering, ConstantOpLowering, FloatProductLowering, FLoatSumLowering>(context);
      patterns.insert<GaussionOpLowering>(context);
      patterns.insert<SingleJointLowering>(typeConverter, context);
      patterns.insert<BatchJointLowering>(typeConverter, context);
      patterns.insert<BatchVectorizeJointLowering>(typeConverter, context, 5);
    }

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDPATTERNS_H
