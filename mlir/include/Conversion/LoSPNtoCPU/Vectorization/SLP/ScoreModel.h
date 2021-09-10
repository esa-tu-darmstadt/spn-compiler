//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SCOREMODEL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SCOREMODEL_H

#include "SLPGraph.h"
#include "GraphConversion.h"
#include "PatternVisitors.h"
#include "SLPPatternMatch.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class ScoreModel {
        public:
          explicit ScoreModel(unsigned lookAhead);
          virtual ~ScoreModel() = default;
          virtual Value getBest(Value value, ArrayRef<Value> candidates) = 0;
        protected:
          unsigned lookAhead;
        };

        class PorpodasModel : public ScoreModel {
          using ScoreModel::ScoreModel;
        public:
          Value getBest(Value value, ArrayRef<Value> candidates) override;
        private:
          unsigned getLookAheadScore(Value last, Value candidate, unsigned maxLevel) const;
        };

        class XorChainModel : public ScoreModel {
          using ScoreModel::ScoreModel;
        public:
          Value getBest(Value value, ArrayRef<Value> candidates) override;
        private:

          void computeBitCodes(Block* block);

          struct BitCodeMap {
            void encode(Value value);
            llvm::BitVector const& lookup(Value value) const;
            size_t size() const;
            size_t codeWidth() const;
            bool empty() const;
            void dump() const;
          private:
            llvm::StringMap<size_t> mapping;
            llvm::SmallVector<llvm::BitVector> bitCodes;
          };

          struct LoadIndex {
            LoadIndex(Value batchMem, Value batchIndex, uint32_t sampleIndex);
            bool consecutive(LoadIndex const& rhs) const;
            bool gatherable(LoadIndex const& rhs) const;
            bool operator==(LoadIndex const& rhs) const;
            void dump() const;
          private:
            Value batchMem;
            Value batchIndex;
            uint32_t sampleIndex;
          };

          struct XorChain {
            XorChain(Value value, unsigned lookAhead, BitCodeMap const& encodings);
            unsigned computePenalty(XorChain const& rhs);
            void dump() const;
          private:
            /// Stores the chain as a sequence of operation codes.
            // Since each operation encoding contains many zeros and only a single bit that is 1, llvm::SparseBitVector
            // would be extraordinarily useful. Unfortunately, these do not support XOR operations.
            llvm::BitVector sequence;
            SmallVector<LoadIndex, 8> loads;
          };

          BitCodeMap encodings;
          DenseMap<Value, XorChain> cachedChains;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SCOREMODEL_H
