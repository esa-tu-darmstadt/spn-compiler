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

        class LeafVectorizationPattern : public virtual SLPVectorizationPattern {
          using SLPVectorizationPattern::SLPVectorizationPattern;
        public:
          virtual SmallVector<Value, 4> possiblyRequiredExtractions(Superword* superword) const = 0;
        };

        template<typename SourceOp>
        class OpSpecificVectorizationPattern : public virtual SLPVectorizationPattern {
          using SLPVectorizationPattern::SLPVectorizationPattern;
        public:
          LogicalResult match(Superword* superword) const override {
            bool checkedOperands = false;
            for (auto const& value : *superword) {
              SourceOp op = value.getDefiningOp<SourceOp>();
              // Pattern only applicable to uniform superwords of type SourceOp.
              if (!op) {
                return failure();
              }
              if (!checkedOperands) {
                if (superword->numOperands() != op->getNumOperands()) {
                  return failure();
                }
                checkedOperands = true;
              }
            }
            return success();
          }
        };

        struct BroadcastSuperword : public LeafVectorizationPattern {
          using LeafVectorizationPattern::LeafVectorizationPattern;
          LogicalResult match(Superword* superword) const override;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
          SmallVector<Value, 4> possiblyRequiredExtractions(Superword* superword) const override;
        };

        struct BroadcastInsertSuperword : public LeafVectorizationPattern {
          using LeafVectorizationPattern::LeafVectorizationPattern;
          LogicalResult match(Superword* superword) const override;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
          SmallVector<Value, 4> possiblyRequiredExtractions(Superword* superword) const override;
        };

        struct VectorizeConstant : public OpSpecificVectorizationPattern<ConstantOp>, public LeafVectorizationPattern {
          using OpSpecificVectorizationPattern<ConstantOp>::OpSpecificVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
          SmallVector<Value, 4> possiblyRequiredExtractions(Superword* superword) const override;
        };

        struct VectorizeBatchRead : public OpSpecificVectorizationPattern<SPNBatchRead> {
          using OpSpecificVectorizationPattern<SPNBatchRead>::OpSpecificVectorizationPattern;
          LogicalResult match(Superword* superword) const override;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        struct VectorizeAdd : public OpSpecificVectorizationPattern<SPNAdd> {
          using OpSpecificVectorizationPattern<SPNAdd>::OpSpecificVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        struct VectorizeMul : public OpSpecificVectorizationPattern<SPNMul> {
          using OpSpecificVectorizationPattern<SPNMul>::OpSpecificVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        struct VectorizeGaussian : public OpSpecificVectorizationPattern<SPNGaussianLeaf> {
          using OpSpecificVectorizationPattern<SPNGaussianLeaf>::OpSpecificVectorizationPattern;
          void rewrite(Superword* superword, PatternRewriter& rewriter) const override;
          void accept(PatternVisitor& visitor, Superword* superword) override;
        };

        // TODO: add vector reduction patterns

        class PatternVisitor {
        public:
          virtual void visit(BroadcastSuperword* pattern, Superword* superword) = 0;
          virtual void visit(BroadcastInsertSuperword* pattern, Superword* superword) = 0;
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
          patterns.emplace_back(std::make_unique<BroadcastSuperword>(conversionManager));
          patterns.emplace_back(std::make_unique<BroadcastInsertSuperword>(conversionManager));
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
