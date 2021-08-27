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
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {

    struct VectorizeBatchTask : OpConversionPattern<low::SPNTask> {

      using OpConversionPattern<low::SPNTask>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNTask op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNCPUVectorizationStructurePatterns(OwningRewritePatternList& patterns,
                                                               MLIRContext* context,
                                                               TypeConverter& typeConverter) {
      patterns.insert<VectorizeBatchTask>(typeConverter, context, 5);
    }

    struct VectorizeTransposedBatchRead : public OpConversionPattern<low::SPNBatchRead> {

      using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchRead op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeBatchRead : public OpConversionPattern<low::SPNBatchRead> {

      using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchRead op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeBatchWrite : public OpConversionPattern<low::SPNBatchWrite> {

      using OpConversionPattern<low::SPNBatchWrite>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchWrite op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeMul : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogMul : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeAdd : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogAdd : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLog : public OpConversionPattern<low::SPNLog> {

      using OpConversionPattern<low::SPNLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNLog op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeGaussian : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogGaussian : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeCategorical : public OpConversionPattern<low::SPNCategoricalLeaf> {

      using OpConversionPattern<low::SPNCategoricalLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCategoricalLeaf op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeHistogram : public OpConversionPattern<low::SPNHistogramLeaf> {

      using OpConversionPattern<low::SPNHistogramLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNHistogramLeaf op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeConstant : public OpConversionPattern<low::SPNConstant> {

      using OpConversionPattern<low::SPNConstant>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConstant op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveVectorizedStripLog : public OpConversionPattern<low::SPNStripLog> {

      using OpConversionPattern<low::SPNStripLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNStripLog op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveVectorizedConvertLog : public OpConversionPattern<low::SPNConvertLog> {

      using OpConversionPattern<low::SPNConvertLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConvertLog op,
                                    ArrayRef <Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNCPUVectorizationNodePatterns(OwningRewritePatternList& patterns,
                                                                 MLIRContext* context,
                                                                 TypeConverter& typeConverter) {
      patterns.insert<VectorizeTransposedBatchRead, VectorizeBatchRead, VectorizeBatchWrite>(typeConverter, context, 2);
      patterns.insert<VectorizeCategorical, VectorizeHistogram>(typeConverter, context, 2);
      patterns.insert<VectorizeGaussian, VectorizeLogGaussian>(typeConverter, context, 2);
      patterns.insert<VectorizeAdd, VectorizeMul, VectorizeLog>(typeConverter, context, 2);
      patterns.insert<VectorizeLogAdd, VectorizeLogMul>(typeConverter, context, 2);
      patterns.insert<VectorizeConstant>(typeConverter, context, 2);
      patterns.insert<ResolveVectorizedStripLog, ResolveVectorizedConvertLog>(typeConverter, context, 2);
    }

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTORIZESTRUCTUREPATTERNS_H
