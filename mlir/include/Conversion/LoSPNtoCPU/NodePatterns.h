//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace spn {

struct BatchReadLowering : public OpConversionPattern<low::SPNBatchRead> {

  using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNBatchRead op, low::SPNBatchRead::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct BatchWriteLowering : public OpConversionPattern<low::SPNBatchWrite> {

  using OpConversionPattern<low::SPNBatchWrite>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNBatchWrite op, low::SPNBatchWrite::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CopyLowering : public OpConversionPattern<low::SPNCopy> {

  using OpConversionPattern<low::SPNCopy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNCopy op, low::SPNCopy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ConstantLowering : public OpConversionPattern<low::SPNConstant> {

  using OpConversionPattern<low::SPNConstant>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNConstant op, low::SPNConstant::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ReturnLowering : public OpConversionPattern<low::SPNReturn> {

  using OpConversionPattern<low::SPNReturn>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNReturn op, low::SPNReturn::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LogLowering : public OpConversionPattern<low::SPNLog> {

  using OpConversionPattern<low::SPNLog>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNLog op, low::SPNLog::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct MulLowering : public OpConversionPattern<low::SPNMul> {

  using OpConversionPattern<low::SPNMul>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNMul op, low::SPNMul::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct MulLogLowering : public OpConversionPattern<low::SPNMul> {

  using OpConversionPattern<low::SPNMul>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNMul op, low::SPNMul::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct AddLowering : public OpConversionPattern<low::SPNAdd> {

  using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNAdd op, low::SPNAdd::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct AddLogLowering : public OpConversionPattern<low::SPNAdd> {

  using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNAdd op, low::SPNAdd::Adaptor a,
                  ConversionPatternRewriter &rewriter) const override;
};

struct GaussianLowering : public OpConversionPattern<low::SPNGaussianLeaf> {

  using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNGaussianLeaf op,
                  low::SPNGaussianLeaf::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct GaussianLogLowering : public OpConversionPattern<low::SPNGaussianLeaf> {

  using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNGaussianLeaf op,
                  low::SPNGaussianLeaf::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct HistogramLowering : public OpConversionPattern<low::SPNHistogramLeaf> {

  using OpConversionPattern<low::SPNHistogramLeaf>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNHistogramLeaf op,
                  low::SPNHistogramLeaf::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CategoricalLowering
    : public OpConversionPattern<low::SPNCategoricalLeaf> {

  using OpConversionPattern<low::SPNCategoricalLeaf>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNCategoricalLeaf op,
                  low::SPNCategoricalLeaf::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ResolveConvertToVector
    : public OpConversionPattern<low::SPNConvertToVector> {

  using OpConversionPattern<low::SPNConvertToVector>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNConvertToVector op,
                  low::SPNConvertToVector::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ResolveStripLog : public OpConversionPattern<low::SPNStripLog> {

  using OpConversionPattern<low::SPNStripLog>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNStripLog op, low::SPNStripLog::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ResolveConvertLog : public OpConversionPattern<low::SPNConvertLog> {

  using OpConversionPattern<low::SPNConvertLog>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNConvertLog op, low::SPNConvertLog::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

static inline void
populateLoSPNtoCPUNodePatterns(RewritePatternSet &patterns,
                               MLIRContext *context,
                               TypeConverter &typeConverter) {
  patterns.insert<BatchReadLowering, BatchWriteLowering, CopyLowering>(
      typeConverter, context);
  patterns.insert<LogLowering, ReturnLowering, ConstantLowering>(typeConverter,
                                                                 context);
  patterns.insert<MulLowering, AddLowering>(typeConverter, context);
  patterns.insert<MulLogLowering, AddLogLowering>(typeConverter, context);
  patterns.insert<CategoricalLowering, HistogramLowering>(typeConverter,
                                                          context);
  patterns.insert<GaussianLowering, GaussianLogLowering>(typeConverter,
                                                         context);
  patterns.insert<ResolveConvertToVector, ResolveStripLog, ResolveConvertLog>(
      typeConverter, context);
}
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_NODEPATTERNS_H
