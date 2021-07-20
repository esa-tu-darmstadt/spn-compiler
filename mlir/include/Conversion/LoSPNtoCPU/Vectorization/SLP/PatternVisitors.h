//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_PATTERNVISITORS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_PATTERNVISITORS_H

#include "SLPGraph.h"
#include "SLPVectorizationPatterns.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class PatternVisitor {
        public:
          // Default visiting method so that we don't have to override every single visit method.
          virtual void visitDefault(SLPVectorizationPattern const* pattern, Superword* superword) = 0;
          // Individual pattern visits.
          virtual void visit(BroadcastSuperword const* pattern, Superword* superword);
          virtual void visit(BroadcastInsertSuperword const* pattern, Superword* superword);
          virtual void visit(ShuffleTwoSuperwords const* pattern, Superword* superword);
          virtual void visit(VectorizeConstant const* pattern, Superword* superword);
          virtual void visit(VectorizeSPNConstant const* pattern, Superword* superword);
          virtual void visit(CreateConsecutiveLoad const* pattern, Superword* superword);
          virtual void visit(CreateGatherLoad const* pattern, Superword* superword);
          virtual void visit(VectorizeAdd const* pattern, Superword* superword);
          virtual void visit(VectorizeMul const* pattern, Superword* superword);
          virtual void visit(VectorizeGaussian const* pattern, Superword* superword);
          virtual void visit(VectorizeLogAdd const* pattern, Superword* superword);
          virtual void visit(VectorizeLogMul const* pattern, Superword* superword);
          virtual void visit(VectorizeLogGaussian const* pattern, Superword* superword);
        protected:
          virtual ~PatternVisitor() = default;
        };

        class LeafPatternVisitor : public PatternVisitor {
        public:
          ArrayRef<Value> getRequiredScalarValues(SLPVectorizationPattern const* pattern, Superword* superword);
          void visitDefault(SLPVectorizationPattern const* pattern, Superword* superword) override;
          void visit(BroadcastSuperword const* pattern, Superword* superword) override;
          void visit(BroadcastInsertSuperword const* pattern, Superword* superword) override;
        private:
          SmallVector<Value, 4> scalarValues;
        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_PATTERNVISITORS_H
