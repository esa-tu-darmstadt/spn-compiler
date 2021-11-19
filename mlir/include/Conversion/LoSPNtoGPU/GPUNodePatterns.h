//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_NODEPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_NODEPATTERNS_H

#include <LoSPN/LoSPNAttributes.h>
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/Debug.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

namespace mlir {
  namespace spn {

    struct BatchReadGPULowering : public OpConversionPattern<low::SPNBatchRead> {

      using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchRead op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct BatchWriteGPULowering : public OpConversionPattern<low::SPNBatchWrite> {

      using OpConversionPattern<low::SPNBatchWrite>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchWrite op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct CopyGPULowering : public OpConversionPattern<low::SPNCopy> {

      using OpConversionPattern<low::SPNCopy>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCopy op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ConstantGPULowering : public OpConversionPattern<low::SPNConstant> {

      using OpConversionPattern<low::SPNConstant>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConstant op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ReturnGPULowering : public OpConversionPattern<low::SPNReturn> {

      using OpConversionPattern<low::SPNReturn>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNReturn op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct LogGPULowering : public OpConversionPattern<low::SPNLog> {

      using OpConversionPattern<low::SPNLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNLog op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct MulGPULowering : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct MulLogGPULowering : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct AddGPULowering : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct AddLogGPULowering : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct GaussianGPULowering : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct GaussianLogGPULowering : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct CategoricalGPULowering : public OpConversionPattern<low::SPNCategoricalLeaf> {

      using OpConversionPattern<low::SPNCategoricalLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCategoricalLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct HistogramGPULowering : public OpConversionPattern<low::SPNHistogramLeaf> {

      using OpConversionPattern<low::SPNHistogramLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNHistogramLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;

    private:

      Value processBuckets(llvm::ArrayRef<low::Bucket> buckets, ConversionPatternRewriter& rewriter,
                           Value indexVal, Value defaultVal, Location loc, bool computesLog) const;
    };

    struct ResolveStripLogGPU : public OpConversionPattern<low::SPNStripLog> {

      using OpConversionPattern<low::SPNStripLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNStripLog op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveConvertLogGPU : public OpConversionPattern<low::SPNConvertLog> {

      using OpConversionPattern<low::SPNConvertLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConvertLog op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNtoGPUNodePatterns(OwningRewritePatternList& patterns, MLIRContext* context,
                                                      TypeConverter& typeConverter) {
      patterns.insert<BatchReadGPULowering, BatchWriteGPULowering, CopyGPULowering>(typeConverter, context);
      patterns.insert<LogGPULowering, ReturnGPULowering, ConstantGPULowering>(typeConverter, context);
      patterns.insert<MulGPULowering, AddGPULowering>(typeConverter, context);
      patterns.insert<MulLogGPULowering, AddLogGPULowering>(typeConverter, context);
      patterns.insert<GaussianGPULowering, GaussianLogGPULowering>(typeConverter, context);
      patterns.insert<CategoricalGPULowering, HistogramGPULowering>(typeConverter, context);
      patterns.insert<ResolveStripLogGPU, ResolveConvertLogGPU>(typeConverter, context);
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_NODEPATTERNS_H
