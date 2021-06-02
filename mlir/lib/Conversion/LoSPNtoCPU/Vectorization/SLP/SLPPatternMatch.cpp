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
    : costModel{std::move(costModel)}, patterns{std::move(patterns)},
      leafVisitor{std::make_unique<LeafPatternVisitor>()} {}

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

SLPVectorizationPattern* SLPPatternApplicator::bestMatchIfLeaf(Superword* superword) const {
  double bestCost = 0;
  SLPVectorizationPattern* bestPattern = nullptr;
  for (size_t i = 0; i < patterns.size(); ++i) {
    if (leafVisitor->isLeafPattern(patterns[i].get()) && succeeded(patterns[i]->match(superword))) {
      auto cost = costModel->getSuperwordCost(superword, patterns[i].get());
      if (i == 0 || cost < bestCost) {
        bestCost = cost;
        bestPattern = patterns[i].get();
      }
    }
  }
  return bestPattern;
}

void SLPPatternApplicator::matchAndRewrite(Superword* superword, PatternRewriter& rewriter) {
  auto* pattern = bestMatch(superword);
  if (!pattern) {
    llvm_unreachable("could not apply any pattern to superword. did you forget to add a default pattern?");
  }
  pattern->rewriteSuperword(superword, rewriter);
  bestMatches.erase(superword);
}
