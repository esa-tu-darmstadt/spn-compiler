//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H

#include "SLPNode.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        template<typename SourceOp>
        class SLPVectorizationPattern : public OpRewritePattern<SourceOp> {

        public:
          SLPVectorizationPattern(MLIRContext* context,
                                  PatternBenefit benefit,
                                  llvm::DenseMap<Operation*, SLPNode*> const& parentNodes,
                                  llvm::DenseMap<SLPNode*, llvm::SmallVector<Operation*>>& vectorsByNode)
              : OpRewritePattern<SourceOp>{
              context, benefit}, parentNodes{parentNodes}, vectorsByNode{vectorsByNode} {}

        protected:
          llvm::DenseMap<Operation*, SLPNode*> const& parentNodes;
          llvm::DenseMap<SLPNode*, llvm::SmallVector<Operation*>>& vectorsByNode;
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
                                                     llvm::DenseMap<Operation*, SLPNode*> const& parentNodes,
                                                     llvm::DenseMap<SLPNode*,
                                                                    llvm::SmallVector<Operation*>>& vectorsByNode) {
          patterns.insert<VectorizeConstant>(context, 2, parentNodes, vectorsByNode);
          patterns.insert<VectorizeBatchRead>(context, 2, parentNodes, vectorsByNode);
          patterns.insert<VectorizeAdd, VectorizeMul>(context, 2, parentNodes, vectorsByNode);
          patterns.insert<VectorizeGaussian>(context, 2, parentNodes, vectorsByNode);
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
