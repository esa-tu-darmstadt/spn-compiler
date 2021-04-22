//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H

#include <utility>

#include "SLPNode.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Rewrite/FrozenRewritePatternList.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        template<typename SourceOp>
        class SLPVectorizationPattern : public OpConversionPattern<SourceOp> {

        public:
          SLPVectorizationPattern(TypeConverter& typeConverter,
                                  MLIRContext* context,
                                  NodeVector* const& vector,
                                  DenseMap<NodeVector*, Operation*>& vectorizedOps,
                                  SmallPtrSet<NodeVector*, 32>& erasableOps) : OpConversionPattern<SourceOp>{context},
                                                                               vector{vector},
                                                                               vectorizedOps{vectorizedOps},
                                                                               erasableOps{erasableOps} {}

        protected:
          /// The current vector being transformed.
          NodeVector* const& vector;
          /// Vector operations created for the SLP node vectors.
          DenseMap<NodeVector*, Operation*>& vectorizedOps;

          SmallPtrSet<NodeVector*, 32>& erasableOps;
        };

        struct VectorizeConstant : public SLPVectorizationPattern<ConstantOp> {
          using SLPVectorizationPattern<ConstantOp>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(ConstantOp constantOp,
                                        ArrayRef<Value> operands,
                                        ConversionPatternRewriter& rewriter) const override;
        };

        struct VectorizeBatchRead : public SLPVectorizationPattern<SPNBatchRead> {
          using SLPVectorizationPattern<SPNBatchRead>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(SPNBatchRead batchReadOp,
                                        ArrayRef<Value> operands,
                                        ConversionPatternRewriter& rewriter) const override;
        };

        struct VectorizeAdd : public SLPVectorizationPattern<SPNAdd> {
          using SLPVectorizationPattern<SPNAdd>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(SPNAdd addOp,
                                        ArrayRef<Value> operands,
                                        ConversionPatternRewriter& rewriter) const override;
        };

        struct VectorizeMul : public SLPVectorizationPattern<SPNMul> {
          using SLPVectorizationPattern<SPNMul>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(SPNMul mulOp,
                                        ArrayRef<Value> operands,
                                        ConversionPatternRewriter& rewriter) const override;
        };

        struct VectorizeGaussian : public SLPVectorizationPattern<SPNGaussianLeaf> {
          using SLPVectorizationPattern<SPNGaussianLeaf>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(SPNGaussianLeaf gaussianOp,
                                        ArrayRef<Value> operands,
                                        ConversionPatternRewriter& rewriter) const override;
        };

        static void populateSLPVectorizationPatterns(OwningRewritePatternList& patterns,
                                                     TypeConverter& typeConverter,
                                                     MLIRContext* context,
                                                     NodeVector* const& vector,
                                                     DenseMap<NodeVector*, Operation*>& vectorizedOps,
                                                     SmallPtrSet<NodeVector*, 32>& erasableOps) {
          patterns.insert<VectorizeConstant>(typeConverter, context, vector, vectorizedOps, erasableOps);
          patterns.insert<VectorizeBatchRead>(typeConverter, context, vector, vectorizedOps, erasableOps);
          patterns.insert<VectorizeAdd, VectorizeMul>(typeConverter, context, vector, vectorizedOps, erasableOps);
          patterns.insert<VectorizeGaussian>(typeConverter, context, vector, vectorizedOps, erasableOps);
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
