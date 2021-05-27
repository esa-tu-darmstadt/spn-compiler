//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H

#include "SLPGraph.h"
#include "GraphConversion.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class PatternVisitor;

        class Visitable {
        public:
          virtual void accept(PatternVisitor& visitor, Superword* superword) = 0;
        protected:
          virtual ~Visitable() = default;
        };

        class SLPVectorizationPattern : public Visitable {
        public:
          explicit SLPVectorizationPattern(ConversionManager& conversionManager);
          void rewriteSuperword(Superword* superword, PatternRewriter& rewriter);
          virtual LogicalResult match(Superword* superword) const = 0;
        protected:
          virtual void rewrite(Superword* superword, PatternRewriter& rewriter) const = 0;
          ConversionManager& conversionManager;
        };

        template<typename SourceOp>
        class OpSpecificSLPVectorizationPattern : public SLPVectorizationPattern {
        public:

          explicit OpSpecificSLPVectorizationPattern(ConversionManager& conversionManager) : SLPVectorizationPattern{
              conversionManager} {}

          LogicalResult match(Superword* superword) const override {
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
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        struct VectorizeBatchRead : public OpSpecificSLPVectorizationPattern<SPNBatchRead> {
          using OpSpecificSLPVectorizationPattern<SPNBatchRead>::OpSpecificSLPVectorizationPattern;
          LogicalResult match(Superword* superword) const override;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        struct VectorizeAdd : public OpSpecificSLPVectorizationPattern<SPNAdd> {
          using OpSpecificSLPVectorizationPattern<SPNAdd>::OpSpecificSLPVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        struct VectorizeMul : public OpSpecificSLPVectorizationPattern<SPNMul> {
          using OpSpecificSLPVectorizationPattern<SPNMul>::OpSpecificSLPVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        struct VectorizeGaussian : public OpSpecificSLPVectorizationPattern<SPNGaussianLeaf> {
          using OpSpecificSLPVectorizationPattern<SPNGaussianLeaf>::OpSpecificSLPVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        class PatternVisitor {
        public:
          virtual void visit(VectorizeConstant* pattern, Superword* superword) = 0;
          virtual void visit(VectorizeBatchRead* pattern, Superword* superword) = 0;
          virtual void visit(VectorizeAdd* pattern, Superword* superword) = 0;
          virtual void visit(VectorizeMul* pattern, Superword* superword) = 0;
          virtual void visit(VectorizeGaussian* pattern, Superword* superword) = 0;
        protected:
          virtual ~PatternVisitor() = default;
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
