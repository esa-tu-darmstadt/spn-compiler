//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPPATTERNMATCH_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPPATTERNMATCH_H

#include "SLPVectorizationPatterns.h"
#include "PatternVisitors.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        /// This class applies rewrite patterns to SLP vectors and effectively creates corresponding SIMD operations. It
        /// automatically determines the best pattern to apply, e.g. based on a cost model.
        class SLPPatternApplicator {
        public:
          SLPPatternApplicator() = default;
          virtual ~SLPPatternApplicator() = default;
          /// Determine the best pattern for the given superword and rewrite it according to the rewrite pattern using
          /// the provided rewriter.
          void matchAndRewrite(Superword* superword, RewriterBase& rewriter) const;
          /// Determine the best pattern for the given superword.
          virtual SLPVectorizationPattern* bestMatch(Superword* superword) const = 0;
          /// Tell the pattern applicator which patterns are available for use.
          void setPatterns(SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>&& slpVectorizationPatterns);
        protected:
          SmallVector<std::unique_ptr<SLPVectorizationPattern>> patterns;
        };

        /// A cost model based pattern applicator chooses SLP vectorization patterns based on the SLP vectorization's
        /// cost model. It chooses the patterns that (according to the cost model) will result in the most profitable
        /// program.
        template<typename CostModel>
        class CostModelPatternApplicator : public SLPPatternApplicator {

        public:

          CostModelPatternApplicator() : SLPPatternApplicator{}, costModel{std::make_unique<CostModel>(*this)} {}

          SLPVectorizationPattern* bestMatch(Superword* superword) const override {
            SLPVectorizationPattern* bestPattern = nullptr;
            double bestCost = 0;
            for (auto const& pattern : patterns) {
              if (succeeded(pattern->match(superword))) {
                auto cost = costModel->getSuperwordCost(superword, pattern.get());
                if (!bestPattern || cost < bestCost) {
                  bestPattern = pattern.get();
                  bestCost = cost;
                }
              }
            }
            return bestPattern;
          }

          CostModel* getCostModel() {
            return costModel.get();
          }

        private:
          std::unique_ptr<CostModel> costModel;
        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPPATTERNMATCH_H
