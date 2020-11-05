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

    struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {

      using OpConversionPattern<ConstantOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    struct ReturnOpLowering : public OpConversionPattern<ReturnOp> {

      using OpConversionPattern<ReturnOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(ReturnOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct SingleJointLowering : public OpConversionPattern<SingleJointQuery> {

      using OpConversionPattern<SingleJointQuery>::OpConversionPattern;

      LogicalResult matchAndRewrite(SingleJointQuery op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

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

    static void populateSPNtoStandardConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                        TypeConverter& typeConverter) {
      patterns.insert<ReturnOpLowering, ConstantOpLowering, FloatProductLowering, FLoatSumLowering>(context);
      patterns.insert<SingleJointLowering>(typeConverter, context);
    }

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDPATTERNS_H