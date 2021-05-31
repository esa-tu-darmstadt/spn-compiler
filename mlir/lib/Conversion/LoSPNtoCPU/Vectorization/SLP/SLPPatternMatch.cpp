//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/SLPPatternMatch.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

SLPPatternApplicator::SLPPatternApplicator(std::shared_ptr<CostModel> costModel,
                                           SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>&& patterns)
    : costModel{std::move(costModel)}, patterns{std::move(patterns)} {}

SLPVectorizationPattern* SLPPatternApplicator::bestMatch(Superword* superword) {
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

void SLPPatternApplicator::matchAndRewrite(Superword* superword, PatternRewriter& rewriter) {
  auto* pattern = bestMatch(superword);
  if (!pattern) {
    llvm_unreachable("could not apply any pattern to superword. did you forget to add a default pattern?");
  }
  pattern->rewriteSuperword(superword, rewriter);
  bestMatches.erase(superword);
}
