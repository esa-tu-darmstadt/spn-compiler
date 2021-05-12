//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H

#include <utility>

#include "SLPGraph.h"
#include "GraphConversion.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternList.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        template<typename SourceOp>
        class SLPVectorizationPattern : public OpRewritePattern<SourceOp> {

        public:
          SLPVectorizationPattern(MLIRContext* context, ValueVector* const& vector, ConversionManager& conversionManager)
              : OpRewritePattern<SourceOp>{context}, vector{vector}, conversionManager{conversionManager} {}

        protected:
          /// The current vector being transformed.
          ValueVector* const& vector;
          ConversionManager& conversionManager;
        };

        struct VectorizeConstant : public SLPVectorizationPattern<ConstantOp> {
          using SLPVectorizationPattern<ConstantOp>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(ConstantOp constantOp, PatternRewriter& rewriter) const override;
        };

        struct VectorizeBatchRead : public SLPVectorizationPattern<SPNBatchRead> {
          using SLPVectorizationPattern<SPNBatchRead>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(SPNBatchRead batchReadOp, PatternRewriter& rewriter) const override;
        };

        struct VectorizeAdd : public SLPVectorizationPattern<SPNAdd> {
          using SLPVectorizationPattern<SPNAdd>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(SPNAdd addOp, PatternRewriter& rewriter) const override;
        };

        struct VectorizeMul : public SLPVectorizationPattern<SPNMul> {
          using SLPVectorizationPattern<SPNMul>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(SPNMul mulOp, PatternRewriter& rewriter) const override;
        };

        struct VectorizeGaussian : public SLPVectorizationPattern<SPNGaussianLeaf> {
          using SLPVectorizationPattern<SPNGaussianLeaf>::SLPVectorizationPattern;
          LogicalResult matchAndRewrite(SPNGaussianLeaf gaussianOp, PatternRewriter& rewriter) const override;
        };

        static void populateSLPVectorizationPatterns(OwningRewritePatternList& patterns,
                                                     MLIRContext* context,
                                                     ValueVector* const& vector,
                                                     ConversionManager& conversionManager) {
          patterns.insert<VectorizeConstant>(context, vector, conversionManager);
          patterns.insert<VectorizeBatchRead>(context, vector, conversionManager);
          patterns.insert<VectorizeAdd, VectorizeMul>(context, vector, conversionManager);
          patterns.insert<VectorizeGaussian>(context, vector, conversionManager);
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
