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

    struct VectorizeTask : OpConversionPattern<low::SPNTask> {

    public:
      VectorizeTask(TypeConverter& typeConverter,
                    MLIRContext* context,
                    PatternBenefit benefit,
                    bool requireAllOpsVectorizable) : OpConversionPattern<low::SPNTask>(typeConverter,
                                                                                        context,
                                                                                        benefit),
                                                      requireAllOpsVectorizable{requireAllOpsVectorizable} {}

    protected:
      LogicalResult createFunctionIfVectorizable(low::SPNTask& task,
                                                 llvm::ArrayRef<Value> const& operands,
                                                 ConversionPatternRewriter& rewriter,
                                                 FuncOp* function) const;

    private:
      using OpConversionPattern<low::SPNTask>::OpConversionPattern;
      bool requireAllOpsVectorizable;
    };

    struct VectorizeSingleTask : public VectorizeTask {
    public:
      VectorizeSingleTask(TypeConverter& typeConverter, MLIRContext* context, PatternBenefit benefit) : VectorizeTask(
          typeConverter,
          context,
          benefit,
          false) {}
    protected:
      LogicalResult matchAndRewrite(low::SPNTask task,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeBatchTask : public VectorizeTask {
    public:
      VectorizeBatchTask(TypeConverter& typeConverter, MLIRContext* context, PatternBenefit benefit) : VectorizeTask(
          typeConverter,
          context,
          benefit,
          true) {}
    protected:
      LogicalResult matchAndRewrite(low::SPNTask op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNtoCPUVectorizationTaskPatterns(OwningRewritePatternList& patterns,
                                                                   MLIRContext* context,
                                                                   TypeConverter& typeConverter) {
      patterns.insert<VectorizeSingleTask, VectorizeBatchTask>(typeConverter, context, 5);
    }

    struct VectorizeBatchRead : public OpConversionPattern<low::SPNBatchRead> {

      using OpConversionPattern<low::SPNBatchRead>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchRead op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeBatchWrite : public OpConversionPattern<low::SPNBatchWrite> {

      using OpConversionPattern<low::SPNBatchWrite>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNBatchWrite op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeMul : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogMul : public OpConversionPattern<low::SPNMul> {

      using OpConversionPattern<low::SPNMul>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNMul op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeAdd : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogAdd : public OpConversionPattern<low::SPNAdd> {

      using OpConversionPattern<low::SPNAdd>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNAdd op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLog : public OpConversionPattern<low::SPNLog> {

      using OpConversionPattern<low::SPNLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNLog op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeGaussian : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeLogGaussian : public OpConversionPattern<low::SPNGaussianLeaf> {

      using OpConversionPattern<low::SPNGaussianLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNGaussianLeaf op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeCategorical : public OpConversionPattern<low::SPNCategoricalLeaf> {

      using OpConversionPattern<low::SPNCategoricalLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNCategoricalLeaf op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeHistogram : public OpConversionPattern<low::SPNHistogramLeaf> {

      using OpConversionPattern<low::SPNHistogramLeaf>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNHistogramLeaf op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct VectorizeConstant : public OpConversionPattern<low::SPNConstant> {

      using OpConversionPattern<low::SPNConstant>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNConstant op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    struct ResolveVectorizedStripLog : public OpConversionPattern<low::SPNStripLog> {

      using OpConversionPattern<low::SPNStripLog>::OpConversionPattern;

      LogicalResult matchAndRewrite(low::SPNStripLog op,
                                    llvm::ArrayRef<Value> operands,
                                    ConversionPatternRewriter& rewriter) const override;
    };

    static inline void populateLoSPNCPUVectorizationNodePatterns(OwningRewritePatternList& patterns,
                                                                 MLIRContext* context,
                                                                 TypeConverter& typeConverter) {
      patterns.insert<VectorizeBatchRead, VectorizeBatchWrite>(typeConverter, context, 2);
      patterns.insert<VectorizeCategorical, VectorizeHistogram>(typeConverter, context, 2);
      patterns.insert<VectorizeGaussian, VectorizeLogGaussian>(typeConverter, context, 2);
      patterns.insert<VectorizeAdd, VectorizeMul, VectorizeLog>(typeConverter, context, 2);
      patterns.insert<VectorizeLogAdd, VectorizeLogMul>(typeConverter, context, 2);
      patterns.insert<VectorizeConstant>(typeConverter, context, 2);
      patterns.insert<ResolveVectorizedStripLog>(typeConverter, context, 2);
    }

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_VECTORIZESTRUCTUREPATTERNS_H
