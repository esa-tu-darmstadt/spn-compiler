//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SCOREMODEL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SCOREMODEL_H

#include "GraphConversion.h"
#include "PatternVisitors.h"
#include "SLPGraph.h"
#include "SLPPatternMatch.h"

namespace mlir {
namespace spn {
namespace low {
namespace slp {

/// A ScoreModel is responsible for assigning scores to candidate operations
/// during the operation reordering step in the SLP graph building phase. Every
/// such model must implement a single method that retrieves a best candidate
/// from a set of candidate operations, depending on the lookahead.
class ScoreModel {
public:
  explicit ScoreModel(unsigned lookAhead);
  virtual ~ScoreModel() = default;
  /// Returns the candidate for the given value that best matches it based on
  /// the model's lookahead.
  virtual Value getBest(Value value, ArrayRef<Value> candidates) = 0;

protected:
  unsigned lookAhead;
};

/// A score model based on Porpodas's Look-Ahead SLP
/// (https://dl.acm.org/doi/10.1145/3168807). Some slight adaptions were done to
/// the score computation such that gather loads and broadcast loads are also
/// taken into account (consecutive score > gather score > broadcast score).
class PorpodasModel : public ScoreModel {
  using ScoreModel::ScoreModel;

public:
  Value getBest(Value value, ArrayRef<Value> candidates) override;

private:
  unsigned getLookAheadScore(Value last, Value candidate,
                             unsigned maxLevel) const;
};

/// A score model that compares operand trees by sorting them, encoding them
/// with bit codes and then comparing them with XOR operations. The more ones
/// there are in the final output, the more different are the operand trees.
/// Also takes into account how many loads there are for pairings and how good
/// they can be paired.
class XorChainModel : public ScoreModel {
  using ScoreModel::ScoreModel;

public:
  Value getBest(Value value, ArrayRef<Value> candidates) override;

private:
  void computeBitCodes(Block *block);

  /// Encodes opcodes with bit sequences.
  struct BitCodeMap {
    void encode(Value value);
    llvm::BitVector const &lookup(Value value) const;
    size_t codeWidth() const;
    bool empty() const;
    void dump() const;

  private:
    llvm::StringMap<size_t> mapping;
    llvm::SmallVector<llvm::BitVector> bitCodes;
  };

  /// Useful for easier comparisons of loads in the XOR chains.
  struct LoadOperation {
    LoadOperation(Value batchMem, Value batchIndex, uint32_t sampleIndex);
    bool consecutive(LoadOperation const &rhs) const;
    bool gatherable(LoadOperation const &rhs) const;
    bool operator==(LoadOperation const &rhs) const;
    void dump() const;

  private:
    Value batchMem;
    Value dynamicIndex;
    uint32_t staticIndex;
  };

  struct XorChain {
    XorChain(Value value, unsigned lookAhead, BitCodeMap const &encodings);
    unsigned computePenalty(XorChain const &rhs);
    void dump() const;

  private:
    /// Stores the chain as a sequence of operation codes.
    // Since each operation encoding contains many zeros and only a single bit
    // that is 1, llvm::SparseBitVector would be extraordinarily useful.
    // Unfortunately, these do not support XOR operations.
    llvm::BitVector sequence;
    SmallVector<LoadOperation, 8> loads;
  };

  BitCodeMap encodings;
  DenseMap<Value, XorChain> cachedChains;
};
} // namespace slp
} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SCOREMODEL_H
