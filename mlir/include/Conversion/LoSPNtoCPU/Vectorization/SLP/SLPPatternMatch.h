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
#include "PatternVisitors.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPPatternApplicator {
        public:
          SLPPatternApplicator(std::shared_ptr<CostModel> costModel,
                               SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>&& patterns);
          SLPVectorizationPattern* bestMatch(Superword* superword);
          SLPVectorizationPattern* bestMatchIfLeaf(Superword* superword) const;
          void matchAndRewrite(Superword* superword, PatternRewriter& rewriter);
        private:
          std::shared_ptr<CostModel> costModel;
          DenseMap<Superword*, SLPVectorizationPattern*> bestMatches;
          SmallVector<std::unique_ptr<SLPVectorizationPattern>> patterns;
          std::unique_ptr<LeafPatternVisitor> const leafVisitor;
        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPPATTERNMATCH_H
