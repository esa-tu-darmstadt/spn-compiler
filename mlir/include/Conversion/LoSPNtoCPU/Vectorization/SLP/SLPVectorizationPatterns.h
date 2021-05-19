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
          SLPVectorizationPattern* bestMatch(ValueVector* vector);
          LogicalResult matchAndRewrite(ValueVector* vector, PatternRewriter& rewriter);
        private:
          DenseMap<ValueVector*, SLPVectorizationPattern*> bestMatches;
          SmallVector<std::unique_ptr<SLPVectorizationPattern>> patterns;
        };

        class SLPVectorizationPattern {

        public:

          SLPVectorizationPattern(ConversionManager& conversionManager, PatternBenefit const& benefit)
              : conversionManager{conversionManager}, benefit{benefit} {}

          virtual unsigned costIfMatches(ValueVector* vector) const {
            return 1;
          }

          virtual LogicalResult matchVector(ValueVector* vector) const = 0;

          void rewriteVector(ValueVector* vector, PatternRewriter& rewriter) {
            conversionManager.setInsertionPointFor(vector);
            rewrite(vector, rewriter);
          }

          PatternBenefit getBenefit() const {
            return benefit;
          }

        protected:
          virtual void rewrite(ValueVector* vector, PatternRewriter& rewriter) const = 0;
          ConversionManager& conversionManager;
        private:
          /// The expected benefit of matching this pattern.
          PatternBenefit const benefit;
        };

        template<typename SourceOp>
        class OpSpecificSLPVectorizationPattern : public SLPVectorizationPattern {
        public:
          OpSpecificSLPVectorizationPattern(ConversionManager& conversionManager, PatternBenefit const& benefit)
              : SLPVectorizationPattern{conversionManager, benefit} {}
          LogicalResult matchVector(ValueVector* vector) const override {
            for (auto const& value : *vector) {
              if (!value.getDefiningOp<SourceOp>()) {
                // Pattern only applicable to uniform vectors of type SourceOp.
                return failure();
              }
            }
            return success();
          }
        };

        struct VectorizeConstant : public OpSpecificSLPVectorizationPattern<ConstantOp> {
          using OpSpecificSLPVectorizationPattern<ConstantOp>::OpSpecificSLPVectorizationPattern;
          unsigned costIfMatches(ValueVector* vector) const override;
          void rewrite(ValueVector* vector, PatternRewriter& rewriter) const override;
        };

        struct VectorizeBatchRead : public OpSpecificSLPVectorizationPattern<SPNBatchRead> {
          using OpSpecificSLPVectorizationPattern<SPNBatchRead>::OpSpecificSLPVectorizationPattern;
          LogicalResult matchVector(ValueVector* vector) const override;
          void rewrite(ValueVector* vector, PatternRewriter& rewriter) const override;
        };

        struct VectorizeAdd : public OpSpecificSLPVectorizationPattern<SPNAdd> {
          using OpSpecificSLPVectorizationPattern<SPNAdd>::OpSpecificSLPVectorizationPattern;
          void rewrite(ValueVector* vector, PatternRewriter& rewriter) const override;
        };

        struct VectorizeMul : public OpSpecificSLPVectorizationPattern<SPNMul> {
          using OpSpecificSLPVectorizationPattern<SPNMul>::OpSpecificSLPVectorizationPattern;
          void rewrite(ValueVector* vector, PatternRewriter& rewriter) const override;
        };

        struct VectorizeGaussian : public OpSpecificSLPVectorizationPattern<SPNGaussianLeaf> {
          using OpSpecificSLPVectorizationPattern<SPNGaussianLeaf>::OpSpecificSLPVectorizationPattern;
          unsigned costIfMatches(ValueVector* vector) const override;
          void rewrite(ValueVector* vector, PatternRewriter& rewriter) const override;
        };

        static void populateSLPVectorizationPatterns(SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>& patterns,
                                                     ConversionManager& conversionManager) {
          patterns.emplace_back(std::make_unique<VectorizeConstant>(conversionManager, 1));
          patterns.emplace_back(std::make_unique<VectorizeBatchRead>(conversionManager, 1));
          patterns.emplace_back(std::make_unique<VectorizeAdd>(conversionManager, 1));
          patterns.emplace_back(std::make_unique<VectorizeMul>(conversionManager, 1));
          patterns.emplace_back(std::make_unique<VectorizeGaussian>(conversionManager, 1));
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
