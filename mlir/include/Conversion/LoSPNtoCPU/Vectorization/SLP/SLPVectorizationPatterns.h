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
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class ConversionManager;
        class PatternVisitor;

        class SLPVectorizationPattern {
        public:
          explicit SLPVectorizationPattern(ConversionManager& conversionManager);
          virtual ~SLPVectorizationPattern() = default;
          void rewriteSuperword(Superword* superword, RewriterBase& rewriter);
          virtual LogicalResult match(Superword* superword) = 0;
          virtual void accept(PatternVisitor& visitor, Superword* superword) const = 0;
        protected:
          virtual Value rewrite(Superword* superword, RewriterBase& rewriter) = 0;
          ConversionManager& conversionManager;
        };

        template<typename SourceOp>
        class OpSpecificVectorizationPattern : public SLPVectorizationPattern {
          using SLPVectorizationPattern::SLPVectorizationPattern;
        public:
          LogicalResult match(Superword* superword) override {
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

        template<typename SourceOp>
        class NormalSpaceVectorizationPattern : public OpSpecificVectorizationPattern<SourceOp> {
          using OpSpecificVectorizationPattern<SourceOp>::OpSpecificVectorizationPattern;
        public:
          LogicalResult match(Superword* superword) override {
            if (failed(OpSpecificVectorizationPattern<SourceOp>::match(superword))) {
              return failure();
            }
            return failure(superword->getElementType().isa<LogType>());
          }
        };

        template<typename SourceOp>
        class LogSpaceVectorizationPattern : public OpSpecificVectorizationPattern<SourceOp> {
          using OpSpecificVectorizationPattern<SourceOp>::OpSpecificVectorizationPattern;
        public:
          LogicalResult match(Superword* superword) override {
            if (failed(OpSpecificVectorizationPattern<SourceOp>::match(superword))) {
              return failure();
            }
            return success(superword->getElementType().isa<LogType>());
          }
        };

        // === Op-agnostic patterns === //

        struct BroadcastSuperword : public SLPVectorizationPattern {
          using SLPVectorizationPattern::SLPVectorizationPattern;
          LogicalResult match(Superword* superword) override;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        struct BroadcastInsertSuperword : public SLPVectorizationPattern {
          using SLPVectorizationPattern::SLPVectorizationPattern;
          LogicalResult match(Superword* superword) override;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        struct ShuffleSuperword : public SLPVectorizationPattern {
          explicit ShuffleSuperword(ConversionManager& conversionManager);
          LogicalResult match(Superword* superword) override;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        private:
          DenseMap<Value, SmallPtrSet<Superword*, 8>> superwordsByValue;
          DenseMap<Superword*, Superword*> shuffleMatches;
        };

        // === Op-specific patterns === //

        struct VectorizeConstant : public OpSpecificVectorizationPattern<ConstantOp> {
          using OpSpecificVectorizationPattern<ConstantOp>::OpSpecificVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        struct VectorizeSPNConstant : public OpSpecificVectorizationPattern<SPNConstant> {
          using OpSpecificVectorizationPattern<SPNConstant>::OpSpecificVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        struct VectorizeBatchRead : public OpSpecificVectorizationPattern<SPNBatchRead> {
          using OpSpecificVectorizationPattern<SPNBatchRead>::OpSpecificVectorizationPattern;
          LogicalResult match(Superword* superword) override;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        // === Op-specific normal space patterns === //

        struct VectorizeAdd : public NormalSpaceVectorizationPattern<SPNAdd> {
          using NormalSpaceVectorizationPattern<SPNAdd>::NormalSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        struct VectorizeMul : public NormalSpaceVectorizationPattern<SPNMul> {
          using NormalSpaceVectorizationPattern<SPNMul>::NormalSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        struct VectorizeGaussian : public NormalSpaceVectorizationPattern<SPNGaussianLeaf> {
          using NormalSpaceVectorizationPattern<SPNGaussianLeaf>::NormalSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        // === Op-specific log-space patterns === //

        struct VectorizeLogAdd : public LogSpaceVectorizationPattern<SPNAdd> {
          using LogSpaceVectorizationPattern<SPNAdd>::LogSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        struct VectorizeLogMul : public LogSpaceVectorizationPattern<SPNMul> {
          using LogSpaceVectorizationPattern<SPNMul>::LogSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        struct VectorizeLogGaussian : public LogSpaceVectorizationPattern<SPNGaussianLeaf> {
          using LogSpaceVectorizationPattern<SPNGaussianLeaf>::LogSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword* superword) const override;
        };

        // TODO: add vector reduction patterns

        static void populateSLPVectorizationPatterns(SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>& patterns,
                                                     ConversionManager& conversionManager) {
          patterns.emplace_back(std::make_unique<BroadcastSuperword>(conversionManager));
          patterns.emplace_back(std::make_unique<BroadcastInsertSuperword>(conversionManager));
          patterns.emplace_back(std::make_unique<ShuffleSuperword>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeConstant>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeSPNConstant>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeBatchRead>(conversionManager));
          // === Normal space patterns === //
          patterns.emplace_back(std::make_unique<VectorizeAdd>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeMul>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeGaussian>(conversionManager));
          // === Log-space patterns === //
          patterns.emplace_back(std::make_unique<VectorizeLogAdd>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeLogMul>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeLogGaussian>(conversionManager));
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
