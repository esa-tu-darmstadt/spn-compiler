//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include "SLPNode.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
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
                                  llvm::DenseMap<Operation*, SLPNode*> const& parentNodes) : OpRewritePattern<SourceOp>{
              context, benefit}, parentNodes{parentNodes} {}

        protected:
          llvm::DenseMap<Operation*, SLPNode*> const& parentNodes;
        };

        struct VectorizeBatchRead : public SLPVectorizationPattern<SPNBatchRead> {

          using SLPVectorizationPattern<SPNBatchRead>::SLPVectorizationPattern;

          LogicalResult matchAndRewrite(SPNBatchRead op, PatternRewriter& rewriter) const override;
        };

        struct VectorizeMul : public SLPVectorizationPattern<SPNMul> {

          using SLPVectorizationPattern<SPNMul>::SLPVectorizationPattern;

          LogicalResult matchAndRewrite(SPNMul op, PatternRewriter& rewriter) const override;
        };

        struct VectorizeAdd : public SLPVectorizationPattern<SPNAdd> {

          using SLPVectorizationPattern<SPNAdd>::SLPVectorizationPattern;

          LogicalResult matchAndRewrite(SPNAdd op, PatternRewriter& rewriter) const override;
        };

        struct VectorizeLog : public SLPVectorizationPattern<SPNLog> {

          using SLPVectorizationPattern<SPNLog>::SLPVectorizationPattern;

          LogicalResult matchAndRewrite(SPNLog op, PatternRewriter& rewriter) const override;
        };

        struct VectorizeConstant : public SLPVectorizationPattern<SPNConstant> {

          using SLPVectorizationPattern<SPNConstant>::SLPVectorizationPattern;

          VectorizeConstant(MLIRContext* context,
                            PatternBenefit benefit,
                            const DenseMap<Operation*, SLPNode*>& parentNodes) : SLPVectorizationPattern{context, benefit, parentNodes} {
            llvm::dbgs() << "Created vectorize constants pattern for SLP vectorization" << "\n";
          }

          LogicalResult matchAndRewrite(SPNConstant op, PatternRewriter& rewriter) const override;
        };

        static void populateSLPVectorizationPatterns(OwningRewritePatternList& patterns,
                                                     MLIRContext* context,
                                                     llvm::DenseMap<Operation*, SLPNode*> const& parentNodes) {
          patterns.insert<VectorizeConstant>(context, 2, parentNodes);
          //patterns.insert<VectorizeBatchRead>(context, 2, parentNodes);
          //patterns.insert<VectorizeAdd, VectorizeMul, VectorizeLog>(context, 2, parentNodes);
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
