//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDPATTERNS_H
#define SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "llvm/Support/Debug.h"

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
    /// Pattern for lowering SPN histogram leaf nodes to the Standard dialect.
    /// This conversion lowers to a global, constant memref representing the histogram values.
    struct HistogramOpLowering : public OpConversionPattern<HistogramOp> {

      using OpConversionPattern<HistogramOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(HistogramOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    ///
    /// Pattern for lowering SPN categorical leaf nodes to the Standard dialect.
    /// This conversion lowers to a global, constant memref representing the category probabilities.
    struct CategoricalOpLowering : public OpConversionPattern<CategoricalOp> {

      using OpConversionPattern<CategoricalOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(CategoricalOp op,
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

    /// Template for patterns lowering SPN n-ary arithmetic operations to Standard dialect.
    /// Will only work if the arithmetic is actually happening on floating-point data types.
    /// \tparam SourceOp SPN dialect operation to lower.
    /// \tparam TargetOp Standard dialect operation to lower to.
    template<typename SourceOp, typename TargetOp>
    class FloatArithmeticOpLowering : public OpConversionPattern<SourceOp> {

      using OpConversionPattern<SourceOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override {
        if (op.getNumOperands() > 2 || operands.size() != op.getNumOperands()) {
          return failure();
        }

        Value firstOperand = operands[0];
        auto firstOperandType = firstOperand.getType();
        auto firstScalarFloat = firstOperandType.template isa<FloatType>();
        auto firstVectorFloat = firstOperandType.template isa<VectorType>() &&
            firstOperandType.template dyn_cast<VectorType>().getElementType().template isa<FloatType>();
        if (!(firstScalarFloat || firstVectorFloat)) {
          // Translate only arithmetic operations operating on floating-point data types
          // or vectors of float.
          return failure();
        }

        Value secondOperand = operands[1];
        auto secondOperandType = secondOperand.getType();
        auto secondScalarFloat = secondOperandType.template isa<FloatType>();
        auto secondVectorFloat = secondOperandType.template isa<VectorType>() &&
            secondOperandType.template dyn_cast<VectorType>().getElementType().template isa<FloatType>();
        if (!(secondScalarFloat || secondVectorFloat)) {
          // Translate only arithmetic operations operating on floating-point data types
          // or vectors of float.
          return failure();
        }

        if (firstVectorFloat && !secondVectorFloat) {
          // The first operand was vectorized, the second not.
          if (secondOperand.getDefiningOp()->template hasTrait<mlir::OpTrait::ConstantLike>()) {
            // The second operand is constant, so we can broadcast it to match the requested vectorization
            // for the first operand.
            secondOperand = rewriter.template create<mlir::vector::BroadcastOp>(op.getLoc(), firstOperandType,
                                                                                secondOperand);
          } else {
            // The first operand was vectorized, the second not and and it is not a constant.
            return mlir::failure();
          }
        } else if (secondVectorFloat && !firstVectorFloat) {
          // The second operand was vectorized, the first not.
          if (firstOperand.getDefiningOp()->template hasTrait<mlir::OpTrait::ConstantLike>()) {
            // The first operand is constant, so we can broadcast it to match the requested vectorization
            // for the second operand.
            firstOperand = rewriter.template create<mlir::vector::BroadcastOp>(op.getLoc(), secondOperandType,
                                                                               firstOperand);
          } else {
            // The second operand was vectorized, the first not and and it is not a constant.
            return mlir::failure();
          }
        }

        rewriter.replaceOpWithNewOp<TargetOp>(op, firstOperand, secondOperand);
        return success();
      }
    };

    using FloatProductLowering = FloatArithmeticOpLowering<ProductOp, mlir::MulFOp>;
    using FLoatSumLowering = FloatArithmeticOpLowering<SumOp, mlir::AddFOp>;

    /// Populate list with all patterns required to lower SPN dialect operations to Standard dialect.
    /// \param patterns Pattern list to fill.
    /// \param context MLIR context.
    /// \param typeConverter Type converter.
    static inline void populateSPNtoStandardConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                        TypeConverter& typeConverter) {
      patterns.insert<ReturnOpLowering, ConstantOpLowering, FloatProductLowering, FLoatSumLowering>(context);
      patterns.insert<HistogramOpLowering, CategoricalOpLowering, GaussionOpLowering>(context);
      patterns.insert<SingleJointLowering>(typeConverter, context);
      patterns.insert<BatchJointLowering>(typeConverter, context);
    }

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDPATTERNS_H
