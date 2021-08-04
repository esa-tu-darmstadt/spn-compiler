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

        class SLPPatternApplicator {
        public:
          SLPPatternApplicator() = default;
          virtual ~SLPPatternApplicator() = default;
          void matchAndRewrite(Superword* superword, RewriterBase& rewriter);
          virtual SLPVectorizationPattern* bestMatch(Superword* superword) const = 0;
          void setPatterns(SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>&& slpVectorizationPatterns);
        protected:
          SmallVector<std::unique_ptr<SLPVectorizationPattern>> patterns;
        };

        template<typename CostModel>
        class CostModelPatternApplicator : public SLPPatternApplicator {

        public:

          CostModelPatternApplicator() : SLPPatternApplicator{}, costModel{std::make_shared<CostModel>(*this)} {}

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

          std::shared_ptr<CostModel> getCostModel() {
            return costModel;
          }

        private:
          std::shared_ptr<CostModel> costModel;
        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPPATTERNMATCH_H
