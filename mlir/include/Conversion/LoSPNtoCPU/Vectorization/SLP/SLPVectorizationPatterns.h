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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class ConversionManager;
        class PatternVisitor;

        /// An SLP vectorization pattern's purpose is describing when it is applicable to a given SLP vector and
        /// rewriting such an SLP vector by creating corresponding SIMD instructions.
        class SLPVectorizationPattern {
        public:
          /// The conversion manager is required for setting up appropriate insertion points and for keeping track of
          /// vector -> SIMD operation mappings.
          explicit SLPVectorizationPattern(ConversionManager& conversionManager);
          virtual ~SLPVectorizationPattern() = default;
          /// Create SIMD instructions for the given superword. Uses the provided rewriter for operation creation.
          /// Also automatically sets up appropriate insertion points.
          void rewriteSuperword(Superword* superword, RewriterBase& rewriter);
          /// Returns true if the pattern is applicable to the provided superword, otherwise false.
          virtual LogicalResult match(Superword* superword) = 0;
          /// Accepts the provided visitor and calls the visitor's method that corresponds to the pattern class.
          virtual void accept(PatternVisitor& visitor, Superword const* superword) const = 0;
        protected:
          virtual Value rewrite(Superword* superword, RewriterBase& rewriter) = 0;
          ConversionManager& conversionManager;
        };

        /// Vectorization patterns that are applicable to specific operations only (additions, ..).
        /// An optional pack of compatible op types can be supplied to create vectors with more than one op code. A
        /// simple use case would be matching vectors containing both MLIR constants and LoSPN constants, as these are
        /// lowered to constants anyways.
        template<typename SourceOp, typename... CompatibleOps>
        class OpSpecificVectorizationPattern : public SLPVectorizationPattern {
          using SLPVectorizationPattern::SLPVectorizationPattern;
        public:
          LogicalResult match(Superword* superword) override {
            for (auto value : *superword) {
              if (auto op = compatibleOperationOrNull<SourceOp, CompatibleOps...>(value)) {
                if (superword->numOperands() == op->getNumOperands()) {
                  continue;
                }
              }
              return failure();
            }
            return success();
          }
        private:

          template<typename LastOp>
          Operation* compatibleOperationOrNull(Value value) {
            if (auto op = value.getDefiningOp<LastOp>()) {
              return op;
            }
            return nullptr;
          }

          template<typename FirstOp, typename SecondOp, typename... Ops>
          Operation* compatibleOperationOrNull(Value value) {
            if (auto op = compatibleOperationOrNull<FirstOp>(value)) {
              return op;
            }
            return compatibleOperationOrNull<SecondOp, Ops...>(value);
          }

        };

        /// Vectorization patterns for computation in normal space.
        template<typename SourceOp, typename... CompatibleOps>
        class NormalSpaceVectorizationPattern : public OpSpecificVectorizationPattern<SourceOp, CompatibleOps...> {
          using OpSpecificVectorizationPattern<SourceOp, CompatibleOps...>::OpSpecificVectorizationPattern;
        public:
          LogicalResult match(Superword* superword) override {
            if (failed(OpSpecificVectorizationPattern<SourceOp, CompatibleOps...>::match(superword))) {
              return failure();
            }
            return failure(superword->getElementType().isa<LogType>());
          }
        };

        /// Vectorization patterns for computation in log space.
        template<typename SourceOp, typename... CompatibleOps>
        class LogSpaceVectorizationPattern : public OpSpecificVectorizationPattern<SourceOp, CompatibleOps...> {
          using OpSpecificVectorizationPattern<SourceOp, CompatibleOps...>::OpSpecificVectorizationPattern;
        public:
          LogicalResult match(Superword* superword) override {
            if (failed(OpSpecificVectorizationPattern<SourceOp, CompatibleOps...>::match(superword))) {
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
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        struct BroadcastInsertSuperword : public SLPVectorizationPattern {
          using SLPVectorizationPattern::SLPVectorizationPattern;
          LogicalResult match(Superword* superword) override;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        struct ShuffleTwoSuperwords : public SLPVectorizationPattern {
          explicit ShuffleTwoSuperwords(ConversionManager& conversionManager);
          LogicalResult match(Superword* superword) override;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        private:
          // For determinism purposes, use a set vector with deterministic iteration order instead of a set.
          DenseMap<Value, llvm::SmallSetVector<Superword*, 8>> superwordsByValue;
          DenseMap<Superword*, std::tuple<Superword*, Superword*, SmallVector<int64_t, 4>>> shuffleMatches;
        };

        // === Op-specific patterns === //

        /// Vectorization pattern for creating SIMD constants.
        struct VectorizeConstant : public OpSpecificVectorizationPattern<arith::ConstantOp, SPNConstant> {
          using OpSpecificVectorizationPattern<arith::ConstantOp, SPNConstant>::OpSpecificVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        /// Vectorization pattern for creating vector load operations.
        struct CreateConsecutiveLoad : public OpSpecificVectorizationPattern<SPNBatchRead> {
          using OpSpecificVectorizationPattern<SPNBatchRead>::OpSpecificVectorizationPattern;
          LogicalResult match(Superword* superword) override;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        /// Vectorization pattern for creating gather load operations.
        struct CreateGatherLoad : public OpSpecificVectorizationPattern<SPNBatchRead> {
          using OpSpecificVectorizationPattern<SPNBatchRead>::OpSpecificVectorizationPattern;
          LogicalResult match(Superword* superword) override;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        // === Op-specific normal space patterns === //

        /// Vectorization pattern for creating normal space vector additions.
        struct VectorizeAdd : public NormalSpaceVectorizationPattern<SPNAdd> {
          using NormalSpaceVectorizationPattern<SPNAdd>::NormalSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        /// Vectorization pattern for creating normal space vector multiplications.
        struct VectorizeMul : public NormalSpaceVectorizationPattern<SPNMul> {
          using NormalSpaceVectorizationPattern<SPNMul>::NormalSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        /// Vectorization pattern for creating vectorized normal distribution evaluations in normal space.
        struct VectorizeGaussian : public NormalSpaceVectorizationPattern<SPNGaussianLeaf> {
          using NormalSpaceVectorizationPattern<SPNGaussianLeaf>::NormalSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        // === Op-specific log-space patterns === //

        /// Vectorization pattern for creating log space vector additions.
        struct VectorizeLogAdd : public LogSpaceVectorizationPattern<SPNAdd> {
          using LogSpaceVectorizationPattern<SPNAdd>::LogSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        /// Vectorization pattern for creating log space vector multiplications.
        struct VectorizeLogMul : public LogSpaceVectorizationPattern<SPNMul> {
          using LogSpaceVectorizationPattern<SPNMul>::LogSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        /// Vectorization pattern for creating vectorized normal distribution evaluations in log space.
        struct VectorizeLogGaussian : public LogSpaceVectorizationPattern<SPNGaussianLeaf> {
          using LogSpaceVectorizationPattern<SPNGaussianLeaf>::LogSpaceVectorizationPattern;
          Value rewrite(Superword* superword, RewriterBase& rewriter) override;
          void accept(PatternVisitor& visitor, Superword const* superword) const override;
        };

        // TODO: add vector reduction patterns
        // TODO: add FMA pattern?

        /// A convenience method that returns all defined vectorization patterns.
        static inline SmallVector<std::unique_ptr<SLPVectorizationPattern>> allSLPVectorizationPatterns(
            ConversionManager& conversionManager) {
          SmallVector<std::unique_ptr<SLPVectorizationPattern>> patterns;
          // === Op-agnostic patterns === //
          patterns.emplace_back(std::make_unique<BroadcastSuperword>(conversionManager));
          patterns.emplace_back(std::make_unique<BroadcastInsertSuperword>(conversionManager));
          patterns.emplace_back(std::make_unique<ShuffleTwoSuperwords>(conversionManager));
          // === Op-specific patterns === //
          patterns.emplace_back(std::make_unique<VectorizeConstant>(conversionManager));
          patterns.emplace_back(std::make_unique<CreateConsecutiveLoad>(conversionManager));
          patterns.emplace_back(std::make_unique<CreateGatherLoad>(conversionManager));
          // === Op-specific normal space patterns === //
          patterns.emplace_back(std::make_unique<VectorizeAdd>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeMul>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeGaussian>(conversionManager));
          // === Op-specific log-space patterns === //
          patterns.emplace_back(std::make_unique<VectorizeLogAdd>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeLogMul>(conversionManager));
          patterns.emplace_back(std::make_unique<VectorizeLogGaussian>(conversionManager));
          // ====================================== //
          return patterns;
        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
