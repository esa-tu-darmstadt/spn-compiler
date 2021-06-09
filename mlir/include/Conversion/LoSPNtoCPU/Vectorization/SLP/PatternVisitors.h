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

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class PatternVisitor;
        /* === Forward declare all patterns here to avoid cyclic dependencies in includes. === */
        class SLPVectorizationPattern;
        struct BroadcastSuperword;
        struct BroadcastInsertSuperword;
        struct VectorizeConstant;
        struct VectorizeBatchRead;
        struct VectorizeAdd;
        struct VectorizeMul;
        struct VectorizeGaussian;
        struct VectorizeLogAdd;
        struct VectorizeLogMul;
        struct VectorizeLogGaussian;
        /* =================================================================================== */

        class Visitable {
        public:
          virtual void accept(PatternVisitor& visitor, Superword* superword) = 0;
        protected:
          virtual ~Visitable() = default;
        };

        class PatternVisitor {
        public:
          // Acts as default visiting method so that we don't have to override every single visit method.
          virtual void visitDefault(SLPVectorizationPattern* pattern, Superword* superword) = 0;
          virtual void visit(BroadcastSuperword* pattern, Superword* superword);
          virtual void visit(BroadcastInsertSuperword* pattern, Superword* superword);
          virtual void visit(VectorizeConstant* pattern, Superword* superword);
          virtual void visit(VectorizeBatchRead* pattern, Superword* superword);
          virtual void visit(VectorizeAdd* pattern, Superword* superword);
          virtual void visit(VectorizeMul* pattern, Superword* superword);
          virtual void visit(VectorizeGaussian* pattern, Superword* superword);
          virtual void visit(VectorizeLogAdd* pattern, Superword* superword);
          virtual void visit(VectorizeLogMul* pattern, Superword* superword);
          virtual void visit(VectorizeLogGaussian* pattern, Superword* superword);
        protected:
          virtual ~PatternVisitor() = default;
        };

        class ScalarValueVisitor : public PatternVisitor {
        public:
          ArrayRef<Value> getRequiredScalarValues(SLPVectorizationPattern* pattern, Superword* superword);
          void visitDefault(SLPVectorizationPattern* pattern, Superword* superword) override;
          void visit(BroadcastSuperword* pattern, Superword* superword) override;
          void visit(BroadcastInsertSuperword* pattern, Superword* superword) override;
        private:
          SmallVector<Value, 4> scalarValues;
        };

        class LeafPatternVisitor : public PatternVisitor {
        public:
          bool isLeafPattern(SLPVectorizationPattern* pattern);
          void visitDefault(SLPVectorizationPattern* pattern, Superword* superword) override;
          void visit(BroadcastSuperword* pattern, Superword* superword) override;
          void visit(BroadcastInsertSuperword* pattern, Superword* superword) override;
          void visit(VectorizeConstant* pattern, Superword* superword) override;
          void visit(VectorizeBatchRead* pattern, Superword* superword) override;
        private:
          bool isLeaf;
        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_PATTERNVISITORS_H
