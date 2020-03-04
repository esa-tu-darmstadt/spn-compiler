//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_LOWERSPNTOSTANDARDPATTERNS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_LOWERING_PATTERNS_LOWERSPNTOSTANDARDPATTERNS_H

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <codegen/mlir/lowering/types/SPNTypeConverter.h>
#include "SPNOperationLowering.h"

namespace mlir {
  namespace spn {

    struct ConstantOpLowering : public OpRewritePattern<spn::ConstantOp> {

      using OpRewritePattern<spn::ConstantOp>::OpRewritePattern;

      PatternMatchResult matchAndRewrite(spn::ConstantOp op, PatternRewriter& rewriter) const final;

    };

    struct ReturnOpLowering : public OpRewritePattern<spn::ReturnOp> {

      using OpRewritePattern<spn::ReturnOp>::OpRewritePattern;

      PatternMatchResult matchAndRewrite(spn::ReturnOp op, PatternRewriter& rewriter) const final;

    };

    struct InputVarLowering : public SPNOpLowering<InputVarOp> {
      using SPNOpLowering<InputVarOp>::SPNOpLowering;

      PatternMatchResult matchAndRewrite(InputVarOp op, ArrayRef<Value> operands,
                                         ConversionPatternRewriter& rewriter) const override;
    };

    struct FunctionLowering : public SPNOpLowering<FuncOp> {
      using SPNOpLowering<FuncOp>::SPNOpLowering;

      PatternMatchResult matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                                         ConversionPatternRewriter& rewriter) const override;

    };

    struct HistogramLowering : public SPNOpLowering<HistogramOp> {
      using SPNOpLowering<HistogramOp>::SPNOpLowering;

      PatternMatchResult matchAndRewrite(HistogramOp op, ArrayRef<Value> operands,
                                         ConversionPatternRewriter& rewriter) const override;

    };

    struct SingleQueryLowering : public SPNOpLowering<SPNSingleQueryOp> {
      using SPNOpLowering<SPNSingleQueryOp>::SPNOpLowering;

      PatternMatchResult matchAndRewrite(SPNSingleQueryOp op, ArrayRef<Value> operands,
                                         ConversionPatternRewriter& rewriter) const override;

    };

    template<typename SourceOp, typename TargetOp>
    class NAryOpLowering : public SPNOpLowering<SourceOp> {
      using SPNOpLowering<SourceOp>::SPNOpLowering;

      PatternMatchResult matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                                         ConversionPatternRewriter& rewriter) const override {
        if (op.getNumOperands() > 2 || operands.size() != op.getNumOperands()) {
          return NAryOpLowering<SourceOp, TargetOp>::matchFailure();
        }

        rewriter.replaceOpWithNewOp<TargetOp>(op, operands[0], operands[1]);
        return NAryOpLowering<SourceOp, TargetOp>::matchSuccess();
      }
    };

    using ProductOpLowering = NAryOpLowering<ProductOp, mlir::MulFOp>;
    using SumOpLowering = NAryOpLowering<SumOp, mlir::AddFOp>;

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
