//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/Debug.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

namespace mlir {
  namespace spn {

    struct BatchReadLowering : public OpConversionPattern<low::SPNBatchRead> {

      using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchRead op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct BatchWriteLowering : public OpConversionPattern<low::SPNBatchWrite> {

      using OpConversionPattern<low::SPNBatchWrite>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchWrite op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct CopyLowering : public OpConversionPattern<low::SPNCopy> {

      using OpConversionPattern<low::SPNCopy>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCopy op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ConstantLowering : public OpConversionPattern<low::SPNConstant> {

      using OpConversionPattern<low::SPNConstant>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConstant op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ReturnLowering : public OpConversionPattern<low::SPNReturn> {

      using OpConversionPattern<low::SPNReturn>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNReturn op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct LogLowering : public OpConversionPattern<low::SPNLog> {

      using OpConversionPattern<low::SPNLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNLog op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct MulLowering : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct MulLogLowering : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct AddLowering : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct AddLogLowering : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct GaussianLowering : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct GaussianLogLowering : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct HistogramLowering : public OpConversionPattern<low::SPNHistogramLeaf> {

      using OpConversionPattern<low::SPNHistogramLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNHistogramLeaf op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct CategoricalLowering : public OpConversionPattern<low::SPNCategoricalLeaf> {

      using OpConversionPattern<low::SPNCategoricalLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCategoricalLeaf op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveConvertToVector : public OpConversionPattern<low::SPNConvertToVector> {

      using OpConversionPattern<low::SPNConvertToVector>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConvertToVector op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveStripLog : public OpConversionPattern<low::SPNStripLog> {

      using OpConversionPattern<low::SPNStripLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNStripLog op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveConvertLog : public OpConversionPattern<low::SPNConvertLog> {

      using OpConversionPattern<low::SPNConvertLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConvertLog op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;

    };

    static inline void populateLoSPNtoCPUNodePatterns(RewritePatternSet& patterns, MLIRContext* context,
                                                      TypeConverter& typeConverter) {
      patterns.add<BatchReadLowering, BatchWriteLowering, CopyLowering>(typeConverter, context);
      patterns.add<LogLowering, ReturnLowering, ConstantLowering>(typeConverter, context);
      patterns.add<MulLowering, AddLowering>(typeConverter, context);
      patterns.add<MulLogLowering, AddLogLowering>(typeConverter, context);
      patterns.add<CategoricalLowering, HistogramLowering>(typeConverter, context);
      patterns.add<GaussianLowering, GaussianLogLowering>(typeConverter, context);
      patterns.add<ResolveConvertToVector, ResolveStripLog, ResolveConvertLog>(typeConverter, context);
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H
