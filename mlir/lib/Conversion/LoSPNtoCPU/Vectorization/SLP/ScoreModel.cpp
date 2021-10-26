//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/ScoreModel.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// === ScoreModel === //

ScoreModel::ScoreModel(unsigned lookAhead) : lookAhead{lookAhead} {}

// === PorpodasModel === //

Value PorpodasModel::getBest(Value value, ArrayRef<Value> candidates) {
  Value best = nullptr;
  // Look-ahead on various levels
  // TODO: when the level is increased, we recompute everything from the level before. that could be changed...
  for (size_t level = 1; level <= lookAhead; ++level) {
    // Best is the candidate with max score
    unsigned bestScore = 0;
    llvm::SmallSet<unsigned, 4> scores;
    for (auto candidate : candidates) {
      // Get the look-ahead score
      unsigned score = getLookAheadScore(value, candidate, level);
      if (scores.empty() || score > bestScore) {
        best = candidate;
        bestScore = score;
        scores.insert(score);
      }
    }
    // If found best at level don't go deeper
    if (best != nullptr && scores.size() > 1) {
      break;
    }
  }
  assert(best && "no best value found");
  return best;
}

unsigned PorpodasModel::getLookAheadScore(Value last, Value candidate, unsigned maxLevel) const {
  auto* lastOp = last.getDefiningOp();
  auto* candidateOp = candidate.getDefiningOp();
  if (!lastOp || !candidateOp) {
    return last == candidate;
  }
  if (lastOp->getName() != candidateOp->getName()) {
    return 0;
  }
  if (auto lhsLoad = dyn_cast<SPNBatchRead>(lastOp)) {
    // We know both operations share the same opcode.
    auto rhsLoad = cast<SPNBatchRead>(candidateOp);
    if (lhsLoad.batchMem() == rhsLoad.batchMem() && lhsLoad.dynamicIndex() == rhsLoad.dynamicIndex()) {
      if (lhsLoad.staticIndex() + 1 == rhsLoad.staticIndex()) {
        // Returning 3 prefers consecutive loads to gather loads and broadcast loads.
        return 3;
      }
      // Returning 2 prefers gather loads to broadcast loads.
      if (lhsLoad.staticIndex() != rhsLoad.staticIndex()) {
        return 2;
      }
      // Broadcast load.
      return 1;
    } else {
      return 0;
    }
  }
  if (maxLevel == 0) {
    return 1;
  }
  unsigned scoreSum = 0;
  for (auto& lastOperand : getOperands(last)) {
    for (auto& candidateOperand : getOperands(candidate)) {
      scoreSum += getLookAheadScore(lastOperand, candidateOperand, maxLevel - 1);
    }
  }
  return scoreSum;
}

// === XorChainModel === //

namespace {
  void dumpBitVector(llvm::BitVector const& bitVector, std::string const& prefix = "", std::string const& suffix = "") {
    if (!prefix.empty()) {
      llvm::dbgs() << prefix;
    }
    for (size_t i = 0; i < bitVector.size(); ++i) {
      llvm::dbgs() << (bitVector.test(i) ? "1" : "0");
    }
    if (!suffix.empty()) {
      llvm::dbgs() << suffix;
    }
  }
}

Value XorChainModel::getBest(Value value, ArrayRef<Value> candidates) {
  assert(!candidates.empty() && "candidates must not be empty");
  if (encodings.empty()) {
    computeBitCodes(value.getParentBlock());
  }
  cachedChains.try_emplace(value, value, lookAhead, encodings);
  SmallVector<unsigned, 8> penalties;
  for (auto candidate : candidates) {
    cachedChains.try_emplace(candidate, candidate, lookAhead, encodings);
    // Retrieve chains *after* insertion because insertion invalidates DenseMap iterators.
    auto& valueChain = cachedChains.find(value)->getSecond();
    auto& candidateChain = cachedChains.find(candidate)->getSecond();
    penalties.emplace_back(valueChain.computePenalty(candidateChain));
  }
  size_t minIndex = 0;
  unsigned minPenalty = penalties[0];
  for (size_t i = 1; i < penalties.size(); ++i) {
    if (penalties[i] < minPenalty) {
      minIndex = i;
      minPenalty = penalties[i];
    }
  }
  return candidates[minIndex];
}

void XorChainModel::computeBitCodes(Block* block) {
  block->walk([&](Operation* op) {
    for (auto result: op->getResults()) {
      encodings.encode(result);
    }
    for (auto value : op->getOperands()) {
      encodings.encode(value);
    }
  });
}

void XorChainModel::BitCodeMap::encode(Value value) {
  std::string name;
  if (auto* definingOp = value.getDefiningOp()) {
    name = definingOp->getName().getStringRef().str();
  } else if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    name = std::to_string(blockArg.getArgNumber());
  } else {
    llvm_unreachable("unsupported value type");
  }
  auto const& it = mapping.try_emplace(name, mapping.size());
  if (!it.second) {
    return;
  }
  for (size_t i = 0; i < bitCodes.size(); ++i) {
    bitCodes[i].resize(mapping.size());
  }
  auto& bitVector = bitCodes.emplace_back(mapping.size());
  bitVector.set(mapping.size() - 1);
}

llvm::BitVector const& XorChainModel::BitCodeMap::lookup(Value value) const {
  if (auto* definingOp = value.getDefiningOp()) {
    return bitCodes[mapping.lookup(definingOp->getName().getStringRef())];
  } else if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    return bitCodes[mapping.lookup(std::to_string(blockArg.getArgNumber()))];
  }
  llvm_unreachable("unsupported value type");
}

size_t XorChainModel::BitCodeMap::size() const {
  return bitCodes.size();
}

size_t XorChainModel::BitCodeMap::codeWidth() const {
  assert(!bitCodes.empty() && "unknown code width: no value has been encoded yet");
  return bitCodes.front().size();
}

bool XorChainModel::BitCodeMap::empty() const {
  return bitCodes.empty();
}

void XorChainModel::BitCodeMap::dump() const {
  for (auto const& entry : mapping) {
    llvm::dbgs() << entry.first();
    dumpBitVector(bitCodes[entry.second], ": ", "\n");
  }
}

XorChainModel::LoadIndex::LoadIndex(Value batchMem, Value batchIndex, uint32_t sampleIndex) : batchMem{batchMem},
                                                                                              batchIndex{batchIndex},
                                                                                              sampleIndex{sampleIndex} {

}

bool XorChainModel::LoadIndex::consecutive(LoadIndex const& rhs) const {
  return batchMem == rhs.batchMem && batchIndex == rhs.batchIndex && sampleIndex + 1 == rhs.sampleIndex;
}

bool XorChainModel::LoadIndex::gatherable(LoadIndex const& rhs) const {
  return batchMem == rhs.batchMem && batchIndex == rhs.batchIndex && sampleIndex != rhs.sampleIndex;
}

bool XorChainModel::LoadIndex::operator==(LoadIndex const& rhs) const {
  return batchMem == rhs.batchMem && batchIndex == rhs.batchIndex && sampleIndex == rhs.sampleIndex;
}

// Helper functions in anonymous namespace.
namespace {
  void dumpBlockArgOrConstantValueOrValue(Value value) {
    if (auto blockArgument = value.dyn_cast<BlockArgument>()) {
      llvm::dbgs() << "BlockArgument" << blockArgument.getArgNumber();
      return;
    } else if (auto* definingOp = value.getDefiningOp()) {
      if (auto constant = dyn_cast<ConstantOp>(definingOp)) {
        if (auto floatAttr = constant.getValue().dyn_cast<FloatAttr>()) {
          llvm::dbgs() << floatAttr.getValueAsDouble();
          return;
        } else if (auto intAttr = constant.getValue().dyn_cast<IntegerAttr>()) {
          llvm::dbgs() << intAttr.getInt();
          return;
        }
      }
    }
    llvm::dbgs() << value;
  }
}

void XorChainModel::LoadIndex::dump() const {
  dumpBlockArgOrConstantValueOrValue(batchMem);
  llvm::dbgs() << "[";
  dumpBlockArgOrConstantValueOrValue(batchIndex);
  llvm::dbgs() << "[" << sampleIndex << "]]";
}

// Helper functions in anonymous namespace.
namespace {

  template<typename ValueIterator>
  bool sameOpcode(ValueIterator begin, ValueIterator end) {
    if (!begin->getDefiningOp()) {
      return false;
    }
    auto const& name = begin->getDefiningOp()->getName();
    ++begin;
    while (begin != end) {
      if (!begin->getDefiningOp() || begin->getDefiningOp()->getName() != name) {
        return false;
      }
      ++begin;
    }
    return true;
  }

  SmallVector<Value, 8> getOperandChain(Value value, unsigned lookAhead) {
    SmallVector<Value, 8> allOperands;
    SmallVector<Value, 2> currentValues{value};
    for (unsigned remainingLookAhead = lookAhead; remainingLookAhead-- > 0;) {
      SmallVector<Value, 4> operands;
      bool sortAll = sameOpcode(currentValues.begin(), currentValues.end())
          && commutative(currentValues.begin(), currentValues.end());
      for (auto currentValue : currentValues) {
        // Skip block arguments.
        if (!currentValue.getDefiningOp()) {
          continue;
        }
        auto currentOperands = getOperands(currentValue);
        if (!sortAll && commutative(currentValue)) {
          sortByOpcode(currentOperands);
        }
        operands.append(currentOperands);
      }
      if (sortAll) {
        sortByOpcode(operands);
      }
      allOperands.append(operands);
      currentValues.swap(operands);
    }
    return allOperands;
  }
}

XorChainModel::XorChain::XorChain(Value value, unsigned lookAhead, BitCodeMap const& encodings) {
  auto operandChain = getOperandChain(value, lookAhead);
  sequence.resize(operandChain.size() * encodings.codeWidth());
  for (size_t i = 0; i < operandChain.size(); ++i) {
    auto const& bitVector = encodings.lookup(operandChain[i]);
    for (auto it = bitVector.set_bits_begin(); it != bitVector.set_bits_end(); ++it) {
      sequence.set((i * encodings.codeWidth()) + *it);
    }
    if (auto* definingOp = operandChain[i].getDefiningOp()) {
      if (auto batchRead = dyn_cast<SPNBatchRead>(definingOp)) {
        loads.emplace_back(batchRead.batchMem(), batchRead.dynamicIndex(), batchRead.staticIndex());
      }
    }
  }
}

unsigned XorChainModel::XorChain::computePenalty(XorChainModel::XorChain const& rhs) {
  // Unfortunately, there is no simple '^' operator. There probably is some better way of doing it, but I don't see it.
  sequence ^= rhs.sequence;
  unsigned penalty = sequence.count();
  // Restore original sequence.
  sequence ^= rhs.sequence;
  // Search for consecutive loads (i=0), then for gathers (i=1), then for broadcasts (i=2).
  SmallPtrSet<size_t, 8> matchedLoads;
  SmallPtrSet<size_t, 8> rhsMatchedLoads;
  for (unsigned i = 0; i < 3 && loads.size() != matchedLoads.size(); ++i) {
    for (unsigned lhsIndex = 0; lhsIndex < loads.size() && loads.size() != matchedLoads.size(); ++lhsIndex) {
      if (matchedLoads.contains(lhsIndex)) {
        continue;
      }
      for (unsigned rhsIndex = 0; rhsIndex < rhs.loads.size(); ++rhsIndex) {
        if (rhsMatchedLoads.contains(rhsIndex)) {
          continue;
        }
        bool matched = false;
        // Consecutive loads.
        if (i == 0 && loads[lhsIndex].consecutive(rhs.loads[rhsIndex])) {
          // Consecutive loads do not impose any penalties.
          matched = true;
        }
          // Gathers.
        else if (i == 1 && loads[lhsIndex].gatherable(rhs.loads[rhsIndex])) {
          penalty += 1;
          matched = true;
        }
          // Broadcasts.
        else if (i == 2 && loads[lhsIndex] == rhs.loads[rhsIndex]) {
          penalty += 2;
          matched = true;
        }
        if (matched) {
          matchedLoads.insert(lhsIndex);
          rhsMatchedLoads.insert(rhsIndex);
          break;
        }
      }
    }
  }
  // No match for a load? Impose high penalties.
  // Use 3 since unmatched loads should be taken into account as worse than consecutive loads, gathers or broadcasts.
  return penalty + (loads.size() - matchedLoads.size() + (rhs.loads.size() - rhsMatchedLoads.size())) * 3;
}

void XorChainModel::XorChain::dump() const {
  dumpBitVector(sequence, "");
  llvm::dbgs() << ", loads: {\n";
  for (size_t i = 0; i < loads.size(); ++i) {
    llvm::dbgs() << "\t";
    loads[i].dump();
    llvm::dbgs() << "\n";
  }
  llvm::dbgs() << "}\n";
}
