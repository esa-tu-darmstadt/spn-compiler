//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/Debug.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

namespace mlir {
  namespace spn {

    struct BatchReadLowering : public OpConversionPattern<low::SPNBatchRead> {

      using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchRead op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct BatchWriteLowering : public OpConversionPattern<low::SPNBatchWrite> {

      using OpConversionPattern<low::SPNBatchWrite>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchWrite op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct CopyLowering : public OpConversionPattern<low::SPNCopy> {

      using OpConversionPattern<low::SPNCopy>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCopy op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ConstantLowering : public OpConversionPattern<low::SPNConstant> {

      using OpConversionPattern<low::SPNConstant>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConstant op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ReturnLowering : public OpConversionPattern<low::SPNReturn> {

      using OpConversionPattern<low::SPNReturn>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNReturn op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct LogLowering : public OpConversionPattern<low::SPNLog> {

      using OpConversionPattern<low::SPNLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNLog op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct MulLowering : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct MulLogLowering : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct AddLowering : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct AddLogLowering : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct GaussianLowering : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct GaussianLogLowering : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct HistogramLowering : public OpConversionPattern<low::SPNHistogramLeaf> {

      using OpConversionPattern<low::SPNHistogramLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNHistogramLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct CategoricalLowering : public OpConversionPattern<low::SPNCategoricalLeaf> {

      using OpConversionPattern<low::SPNCategoricalLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCategoricalLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveConvertToVector : public OpConversionPattern<low::SPNConvertToVector> {

      using OpConversionPattern<low::SPNConvertToVector>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConvertToVector op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveStripLog : public OpConversionPattern<low::SPNStripLog> {

      using OpConversionPattern<low::SPNStripLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNStripLog op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static void populateLoSPNtoCPUNodePatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                               TypeConverter& typeConverter) {
      patterns.insert<BatchReadLowering, BatchWriteLowering, CopyLowering>(typeConverter, context);
      patterns.insert<LogLowering, ReturnLowering, ConstantLowering>(typeConverter, context);
      patterns.insert<MulLowering, AddLowering>(typeConverter, context);
      patterns.insert<MulLogLowering, AddLogLowering>(typeConverter, context);
      patterns.insert<CategoricalLowering, HistogramLowering>(typeConverter, context);
      patterns.insert<GaussianLowering, GaussianLogLowering>(typeConverter, context);
      patterns.insert<ResolveConvertToVector, ResolveStripLog>(typeConverter, context);
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H
