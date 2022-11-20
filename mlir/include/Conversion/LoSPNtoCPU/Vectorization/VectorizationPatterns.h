//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTORIZESTRUCTUREPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTORIZESTRUCTUREPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

namespace mlir {
  namespace spn {

    struct VectorizeTask : OpConversionPattern<low::SPNTask> {

    public:
      //using OpAdaptor = OpConversionPattern<low::SPNTask>::OpAdaptor;

      VectorizeTask(TypeConverter& typeConverter,
                    MLIRContext* context,
                    PatternBenefit benefit,
                    bool requireAllOpsVectorizable) : OpConversionPattern<low::SPNTask>(typeConverter,
                                                                                        context,
                                                                                        benefit),
                                                      requireAllOpsVectorizable{requireAllOpsVectorizable} {}

    protected:
      LogicalResult createFunctionIfVectorizable(low::SPNTask& task,
                                                 ValueRange operands,
                                                 ConversionPatternRewriter& rewriter,
                                                 mlir::func::FuncOp* function) const;

    private:
      using OpConversionPattern<low::SPNTask>::OpConversionPattern;
      bool requireAllOpsVectorizable;
    };

    struct VectorizeSingleTask : public VectorizeTask {
    public:
      using OpAdaptor = VectorizeTask::OpAdaptor;
      VectorizeSingleTask(TypeConverter& typeConverter,
                          MLIRContext* context,
                          PatternBenefit benefit,
                          unsigned maxAttempts,
                          unsigned maxSuccessfulIterations,
                          unsigned maxNodeSize,
                          unsigned maxLookAhead,
                          bool reorderInstructionsDFS,
                          bool allowDuplicateElements,
                          bool allowTopologicalMixing,
                          bool useXorChains) :
          VectorizeTask(typeConverter, context, benefit, false),
          maxAttempts{maxAttempts},
          maxSuccessfulIterations{maxSuccessfulIterations},
          maxNodeSize{maxNodeSize},
          maxLookAhead{maxLookAhead},
          reorderInstructionsDFS{reorderInstructionsDFS},
          allowDuplicateElements{allowDuplicateElements},
          allowTopologicalMixing{allowTopologicalMixing},
          useXorChains{useXorChains} {}

    protected:
      LogicalResult matchAndRewrite(low::SPNTask task,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
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
      VectorizeBatchTask(TypeConverter& typeConverter, MLIRContext* context, PatternBenefit benefit) :
          VectorizeTask(typeConverter, context, benefit, true) {}

    protected:
      LogicalResult matchAndRewrite(low::SPNTask op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNtoCPUVectorizationTaskPatterns(RewritePatternSet& patterns,
                                                                   MLIRContext* context,
                                                                   TypeConverter& typeConverter,
                                                                   unsigned maxAttempts,
                                                                   unsigned maxSuccessfulIterations,
                                                                   unsigned maxNodeSize,
                                                                   unsigned maxLookAhead,
                                                                   bool reorderInstructionsDFS,
                                                                   bool allowDuplicateElements,
                                                                   bool allowTopologicalMixing,
                                                                   bool useXorChains) {
      patterns.insert<VectorizeSingleTask>(typeConverter, context, 5,
                                           maxAttempts,
                                           maxSuccessfulIterations,
                                           maxNodeSize,
                                           maxLookAhead,
                                           reorderInstructionsDFS,
                                           allowDuplicateElements,
                                           allowTopologicalMixing,
                                           useXorChains);
      patterns.insert<VectorizeBatchTask>(typeConverter, context, 5);
    }

    struct VectorizeTransposedBatchRead : public OpConversionPattern<low::SPNBatchRead> {

      using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchRead op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeBatchRead : public OpConversionPattern<low::SPNBatchRead> {

      using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchRead op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeBatchWrite : public OpConversionPattern<low::SPNBatchWrite> {

      using OpConversionPattern<low::SPNBatchWrite>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchWrite op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeMul : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogMul : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeAdd : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogAdd : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLog : public OpConversionPattern<low::SPNLog> {

      using OpConversionPattern<low::SPNLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNLog op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeGaussian : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogGaussian : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeCategorical : public OpConversionPattern<low::SPNCategoricalLeaf> {

      using OpConversionPattern<low::SPNCategoricalLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCategoricalLeaf op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeHistogram : public OpConversionPattern<low::SPNHistogramLeaf> {

      using OpConversionPattern<low::SPNHistogramLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNHistogramLeaf op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeConstant : public OpConversionPattern<low::SPNConstant> {

      using OpConversionPattern<low::SPNConstant>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConstant op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveVectorizedStripLog : public OpConversionPattern<low::SPNStripLog> {

      using OpConversionPattern<low::SPNStripLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNStripLog op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveVectorizedConvertLog : public OpConversionPattern<low::SPNConvertLog> {

      using OpConversionPattern<low::SPNConvertLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConvertLog op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNCPUVectorizationNodePatterns(RewritePatternSet& patterns,
                                                                 MLIRContext* context,
                                                                 TypeConverter& typeConverter) {
      patterns.add<VectorizeTransposedBatchRead, VectorizeBatchRead, VectorizeBatchWrite>(typeConverter, context, 2);
      patterns.add<VectorizeCategorical, VectorizeHistogram>(typeConverter, context, 2);
      patterns.add<VectorizeGaussian, VectorizeLogGaussian>(typeConverter, context, 2);
      patterns.add<VectorizeAdd, VectorizeMul, VectorizeLog>(typeConverter, context, 2);
      patterns.add<VectorizeLogAdd, VectorizeLogMul>(typeConverter, context, 2);
      patterns.add<VectorizeConstant>(typeConverter, context, 2);
      patterns.add<ResolveVectorizedStripLog, ResolveVectorizedConvertLog>(typeConverter, context, 2);
    }

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTORIZESTRUCTUREPATTERNS_H
