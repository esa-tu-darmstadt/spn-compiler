//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/SLPPatternMatch.h"
#include "LoSPNtoCPU/Vectorization/SLP/CostModel.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// === SLPPatternApplicator === //

void SLPPatternApplicator::matchAndRewrite(Superword* superword, RewriterBase& rewriter) const {
  auto* pattern = bestMatch(superword);
  if (!pattern) {
    llvm_unreachable("could not apply any pattern to superword");
  }
  pattern->rewriteSuperword(superword, rewriter);
}

void SLPPatternApplicator::setPatterns(SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>&& slpVectorizationPatterns) {
  this->patterns = std::move(slpVectorizationPatterns);
}
