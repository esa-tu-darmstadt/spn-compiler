//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H

#include <utility>

#include "SLPGraph.h"
#include "GraphConversion.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPVectorizationPattern;

        class SLPPatternApplicator {
        public:
          explicit SLPPatternApplicator(SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>&& patterns);
          SLPVectorizationPattern* bestMatch(Superword* superword);
          LogicalResult matchAndRewrite(Superword* superword, PatternRewriter& rewriter);
        private:
          DenseMap<Superword*, SLPVectorizationPattern*> bestMatches;
          SmallVector<std::unique_ptr<SLPVectorizationPattern>> patterns;
        };

        class SLPVectorizationPattern {

        public:

          explicit SLPVectorizationPattern(ConversionManager& conversionManager) : conversionManager{
              conversionManager} {}

          virtual unsigned getCost() const {
            return 1;
          }

          virtual LogicalResult matchSuperword(Superword* superword) const = 0;

          void rewriteSuperword(Superword* superword, PatternRewriter& rewriter) {
            conversionManager.setInsertionPointFor(superword);
            rewrite(superword, rewriter);
          }

        protected:
          virtual void rewrite(Superword* superword, PatternRewriter& rewriter) const = 0;
          ConversionManager& conversionManager;
        };

        template<typename SourceOp>
        class OpSpecificSLPVectorizationPattern : public SLPVectorizationPattern {
        public:

          explicit OpSpecificSLPVectorizationPattern(ConversionManager& conversionManager) : SLPVectorizationPattern{
              conversionManager} {}

          LogicalResult matchSuperword(Superword* superword) const override {
            for (auto const& value : *superword) {
              if (!value.getDefiningOp<SourceOp>()) {
                // Pattern only applicable to uniform superwords of type SourceOp.
                return failure();
              }
            }
            return success();
          }
        };

        struct VectorizeConstant : public OpSpecificSLPVectorizationPattern<ConstantOp> {
          using OpSpecificSLPVectorizationPattern<ConstantOp>::OpSpecificSLPVectorizationPattern;
          unsigned getCost() const override;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
        };

        struct VectorizeBatchRead : public OpSpecificSLPVectorizationPattern<SPNBatchRead> {
          using OpSpecificSLPVectorizationPattern<SPNBatchRead>::OpSpecificSLPVectorizationPattern;
          LogicalResult matchSuperword(Superword* superword) const override;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
        };

        struct VectorizeAdd : public OpSpecificSLPVectorizationPattern<SPNAdd> {
          using OpSpecificSLPVectorizationPattern<SPNAdd>::OpSpecificSLPVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
        };

        struct VectorizeMul : public OpSpecificSLPVectorizationPattern<SPNMul> {
          using OpSpecificSLPVectorizationPattern<SPNMul>::OpSpecificSLPVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
        };

        struct VectorizeGaussian : public OpSpecificSLPVectorizationPattern<SPNGaussianLeaf> {
          using OpSpecificSLPVectorizationPattern<SPNGaussianLeaf>::OpSpecificSLPVectorizationPattern;
          unsigned getCost() const override;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
        };

        static void populateSLPVectorizationPatterns(SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>& patterns,
                                                     ConversionManager& conversionManager) {
          patterns.emplace_back(std::make_unique<VectorizeConstant>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeBatchRead>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeAdd>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeMul>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeGaussian>(conversionManager));
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
