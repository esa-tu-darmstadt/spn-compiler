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
#include "CostModel.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPPatternApplicator {
        public:

          SLPPatternApplicator(std::shared_ptr<CostModel> costModel,
                               SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>&& patterns) : costModel{
              std::move(costModel)}, patterns{std::move(patterns)} {}

          SLPVectorizationPattern* bestMatch(Superword* superword) {
            auto it = bestMatches.try_emplace(superword, nullptr);
            if (it.second) {
              double bestCost = 0;
              for (auto const& pattern : patterns) {
                if (succeeded(pattern->match(superword))) {
                  auto cost = costModel->getSuperwordCost(superword, pattern.get());
                  if (!it.first->second || cost < bestCost) {
                    it.first->second = pattern.get();
                    bestCost = cost;
                  }
                }
              }
            }
            return it.first->second;
          }

          LogicalResult matchAndRewrite(Superword* superword, PatternRewriter& rewriter) {
            auto* pattern = bestMatch(superword);
            if (!pattern) {
              return failure();
            }
            pattern->rewriteSuperword(superword, rewriter);
            bestMatches.erase(superword);
            return success();
          }

        private:
          std::shared_ptr<CostModel> costModel;
          DenseMap<Superword*, SLPVectorizationPattern*> bestMatches;
          SmallVector<std::unique_ptr<SLPVectorizationPattern>> patterns;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPPATTERNMATCH_H
