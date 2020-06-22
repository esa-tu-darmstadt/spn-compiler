//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_LOWERSPNTOSTANDARDPATTERNS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_LOWERSPNTOSTANDARDPATTERNS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <codegen/mlir/lowering/types/SPNTypeConverter.h>
#include "SPNOperationLowering.h"

namespace mlir {
  namespace spn {

    ///
    /// Pattern to lower a constant operation from the SPN dialect to a
    /// constant operation from the Standard dialect.
    struct ConstantOpLowering : public OpRewritePattern<spn::ConstantOp> {

      using OpRewritePattern<spn::ConstantOp>::OpRewritePattern;

      /// Rewrite the operation if it matches this pattern.
      /// \param op Operation to match.
      /// \param rewriter Rewriter to create and insert operations.
      /// \return Indication if the match was successful.
      LogicalResult matchAndRewrite(spn::ConstantOp op, PatternRewriter& rewriter) const final;

    };

    ///
    /// Pattern to lower a return operation from the SPN dialect to a
    /// return operation from the Standard dialect.
    struct ReturnOpLowering : public OpRewritePattern<spn::ReturnOp> {

      using OpRewritePattern<spn::ReturnOp>::OpRewritePattern;

      /// Rewrite the operation if it matches this pattern.
      /// \param op Operation to match.
      /// \param rewriter Rewriter to create and insert operations.
      /// \return Indication if the match was successful.
      LogicalResult matchAndRewrite(spn::ReturnOp op, PatternRewriter& rewriter) const final;

    };

    ///
    /// Pattern to lower a InputVarOp from the SPN dialect to operations
    /// from the Standard dialect.
    struct InputVarLowering : public SPNOpLowering<InputVarOp> {

      using SPNOpLowering<InputVarOp>::SPNOpLowering;

      LogicalResult matchAndRewrite(InputVarOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    ///
    /// Pattern to rewrite the signature of functions during the
    /// conversion from the SPN dialect to Standard dialect.
    struct FunctionLowering : public SPNOpLowering<FuncOp> {
      using SPNOpLowering<FuncOp>::SPNOpLowering;

      LogicalResult matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    ///
    /// Pattern to lower a HistogramOp from the SPN dialect to operations
    /// from the Standard dialect.
    struct HistogramLowering : public SPNOpLowering<HistogramOp> {
      using SPNOpLowering<HistogramOp>::SPNOpLowering;

      LogicalResult matchAndRewrite(HistogramOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    ///
    /// Pattern to lower a SPNSingleQueryOp from the SPN dialect to operations
    /// from the Standard dialect.
    struct SingleQueryLowering : public SPNOpLowering<SPNSingleQueryOp> {
      using SPNOpLowering<SPNSingleQueryOp>::SPNOpLowering;

      LogicalResult matchAndRewrite(SPNSingleQueryOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    /// Pattern to lower a n-ary operation from the SPN dialect to operations
    /// from the Standard dialect.
    /// Can only be applied to operations from the SPN dialect inheriting from SPN_NAry_Op.
    /// \tparam SourceOp Operation type to lower.
    /// \tparam TargetOp Operation type to generate for this operation.
    template<typename SourceOp, typename TargetOp>
    class NAryOpLowering : public SPNOpLowering<SourceOp> {
      using SPNOpLowering<SourceOp>::SPNOpLowering;

      LogicalResult matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override {
        if (op.getNumOperands() > 2 || operands.size() != op.getNumOperands()) {
          return failure();
        }

        rewriter.replaceOpWithNewOp<TargetOp>(op, operands[0], operands[1]);
        return success();
      }
    };

    using ProductOpLowering = NAryOpLowering<ProductOp, mlir::MulFOp>;
    using SumOpLowering = NAryOpLowering<SumOp, mlir::AddFOp>;

    /// Populate the pattern list with all patterns for lowering SPN dialect operations to the Standard dialect.
    /// \param patterns List of patterns.
    /// \param context Surrounding MLIR context.
    /// \param typeConverter Type converter to use for type conversions.
    static void populateSPNtoStandardConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                        TypeConverter& typeConverter) {
      patterns.insert<ConstantOpLowering>(context);
      patterns.insert<ReturnOpLowering>(context);
      patterns.insert<InputVarLowering>(context, typeConverter);
      patterns.insert<FunctionLowering>(context, typeConverter);
      patterns.insert<HistogramLowering>(context, typeConverter);
      patterns.insert<SingleQueryLowering>(context, typeConverter);
      patterns.insert<ProductOpLowering>(context, typeConverter);
      patterns.insert<SumOpLowering>(context, typeConverter);
    }

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_LOWERSPNTOSTANDARDPATTERNS_H
