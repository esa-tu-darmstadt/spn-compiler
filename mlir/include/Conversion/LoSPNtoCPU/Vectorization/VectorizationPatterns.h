//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTORIZESTRUCTUREPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTORIZESTRUCTUREPATTERNS_H

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace spn {

struct VectorizeTask : OpConversionPattern<low::SPNTask> {
protected:
  unsigned vectorWidth;

public:
  VectorizeTask(TypeConverter &typeConverter, MLIRContext *context,
                PatternBenefit benefit, bool requireAllOpsVectorizable,
                unsigned vectorWidth)
      : OpConversionPattern<low::SPNTask>(typeConverter, context, benefit),
        vectorWidth{vectorWidth},
        requireAllOpsVectorizable{requireAllOpsVectorizable} {
    llvm::outs() << "VectorizeTask: vectorWidth = " << vectorWidth << "\n";
  }

protected:
  LogicalResult createFunctionIfVectorizable(
      low::SPNTask &task, low::SPNTask::Adaptor adaptor,
      ConversionPatternRewriter &rewriter, func::FuncOp *function) const;

private:
  using OpConversionPattern<low::SPNTask>::OpConversionPattern;
  bool requireAllOpsVectorizable;
};

struct VectorizeSingleTask : public VectorizeTask {
public:
  VectorizeSingleTask(TypeConverter &typeConverter, MLIRContext *context,
                      PatternBenefit benefit, unsigned maxAttempts,
                      unsigned maxSuccessfulIterations, unsigned maxNodeSize,
                      unsigned maxLookAhead, bool reorderInstructionsDFS,
                      bool allowDuplicateElements, bool allowTopologicalMixing,
                      bool useXorChains, unsigned vectorWidth)
      : VectorizeTask(typeConverter, context, benefit, false, vectorWidth),
        maxAttempts{maxAttempts},
        maxSuccessfulIterations{maxSuccessfulIterations},
        maxNodeSize{maxNodeSize}, maxLookAhead{maxLookAhead},
        reorderInstructionsDFS{reorderInstructionsDFS},
        allowDuplicateElements{allowDuplicateElements},
        allowTopologicalMixing{allowTopologicalMixing},
        useXorChains{useXorChains} {}

protected:
  LogicalResult
  matchAndRewrite(low::SPNTask task, low::SPNTask::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  unsigned maxAttempts;
  unsigned maxSuccessfulIterations;
  unsigned maxNodeSize;
  unsigned maxLookAhead;
  bool reorderInstructionsDFS;
  bool allowDuplicateElements;
  bool allowTopologicalMixing;
  bool useXorChains;
};

struct VectorizeBatchTask : public VectorizeTask {
public:
  VectorizeBatchTask(TypeConverter &typeConverter, MLIRContext *context,
                     PatternBenefit benefit, unsigned vectorWidth)
      : VectorizeTask(typeConverter, context, benefit, true, vectorWidth) {}

protected:
  LogicalResult
  matchAndRewrite(low::SPNTask op, low::SPNTask::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

static inline void populateLoSPNtoCPUVectorizationTaskPatterns(
    RewritePatternSet &patterns, MLIRContext *context,
    TypeConverter &typeConverter, unsigned maxAttempts,
    unsigned maxSuccessfulIterations, unsigned maxNodeSize,
    unsigned maxLookAhead, bool reorderInstructionsDFS,
    bool allowDuplicateElements, bool allowTopologicalMixing, bool useXorChains,
    unsigned vectorWidth) {
  patterns.insert<VectorizeSingleTask>(
      typeConverter, context, 5, maxAttempts, maxSuccessfulIterations,
      maxNodeSize, maxLookAhead, reorderInstructionsDFS, allowDuplicateElements,
      allowTopologicalMixing, useXorChains, vectorWidth);
  patterns.insert<VectorizeBatchTask>(typeConverter, context, 5, vectorWidth);
}

struct VectorizeTransposedBatchRead
    : public OpConversionPattern<low::SPNBatchRead> {

  using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNBatchRead op, low::SPNBatchRead::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeBatchRead : public OpConversionPattern<low::SPNBatchRead> {

  using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNBatchRead op, low::SPNBatchRead::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeBatchWrite : public OpConversionPattern<low::SPNBatchWrite> {

  using OpConversionPattern<low::SPNBatchWrite>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNBatchWrite op, low::SPNBatchWrite::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeMul : public OpConversionPattern<low::SPNMul> {

  using OpConversionPattern<low::SPNMul>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNMul op, low::SPNMul::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeLogMul : public OpConversionPattern<low::SPNMul> {

  using OpConversionPattern<low::SPNMul>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNMul op, low::SPNMul::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeAdd : public OpConversionPattern<low::SPNAdd> {

  using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNAdd op, low::SPNAdd::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeLogAdd : public OpConversionPattern<low::SPNAdd> {

  using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNAdd op, low::SPNAdd::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeLog : public OpConversionPattern<low::SPNLog> {

  using OpConversionPattern<low::SPNLog>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNLog op, low::SPNLog::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeGaussian : public OpConversionPattern<low::SPNGaussianLeaf> {

  using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNGaussianLeaf op,
                  low::SPNGaussianLeaf::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeLogGaussian : public OpConversionPattern<low::SPNGaussianLeaf> {

  using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNGaussianLeaf op,
                  low::SPNGaussianLeaf::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeCategorical
    : public OpConversionPattern<low::SPNCategoricalLeaf> {

  using OpConversionPattern<low::SPNCategoricalLeaf>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNCategoricalLeaf op,
                  low::SPNCategoricalLeaf::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeHistogram : public OpConversionPattern<low::SPNHistogramLeaf> {

  using OpConversionPattern<low::SPNHistogramLeaf>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNHistogramLeaf op,
                  low::SPNHistogramLeaf::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorizeConstant : public OpConversionPattern<low::SPNConstant> {

  using OpConversionPattern<low::SPNConstant>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNConstant op, low::SPNConstant::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ResolveVectorizedStripLog
    : public OpConversionPattern<low::SPNStripLog> {

  using OpConversionPattern<low::SPNStripLog>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNStripLog op, low::SPNStripLog::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ResolveVectorizedConvertLog
    : public OpConversionPattern<low::SPNConvertLog> {

  using OpConversionPattern<low::SPNConvertLog>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(low::SPNConvertLog op, low::SPNConvertLog::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

static inline void
populateLoSPNCPUVectorizationNodePatterns(RewritePatternSet &patterns,
                                          MLIRContext *context,
                                          TypeConverter &typeConverter) {
  patterns.insert<VectorizeTransposedBatchRead, VectorizeBatchRead,
                  VectorizeBatchWrite>(typeConverter, context, 2);
  patterns.insert<VectorizeCategorical, VectorizeHistogram>(typeConverter,
                                                            context, 2);
  patterns.insert<VectorizeGaussian, VectorizeLogGaussian>(typeConverter,
                                                           context, 2);
  patterns.insert<VectorizeAdd, VectorizeMul, VectorizeLog>(typeConverter,
                                                            context, 2);
  patterns.insert<VectorizeLogAdd, VectorizeLogMul>(typeConverter, context, 2);
  patterns.insert<VectorizeConstant>(typeConverter, context, 2);
  patterns.insert<ResolveVectorizedStripLog, ResolveVectorizedConvertLog>(
      typeConverter, context, 2);
}

} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTORIZESTRUCTUREPATTERNS_H
