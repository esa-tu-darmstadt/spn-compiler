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
  // TODO: when the level is increased, we recompute everything from the level before. change that maybe?
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
    if (lhsLoad.batchMem() == rhsLoad.batchMem() && lhsLoad.batchIndex() == rhsLoad.batchIndex()) {
      if (lhsLoad.sampleIndex() + 1 == rhsLoad.sampleIndex()) {
        // Returning 3 prefers consecutive loads to gather loads and broadcast loads.
        return 3;
      }
      // Returning 2 prefers gather loads to broadcast loads.
      if (lhsLoad.sampleIndex() != rhsLoad.sampleIndex()) {
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
  if (bitMap.empty()) {
    computeBitCodes(value.getParentBlock());
  }
  cachedChains.try_emplace(value, value, lookAhead, bitMap);
  SmallVector<unsigned, 8> penalties;
  for (auto candidate : candidates) {
    cachedChains.try_emplace(candidate, candidate, lookAhead, bitMap);
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
      bitMap.encode(result);
    }
    for (auto value : op->getOperands()) {
      bitMap.encode(value);
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

bool XorChainModel::LoadIndex::consecutive(LoadIndex const& rhs) const {
  return batchMem == rhs.batchMem && batchIndex == rhs.batchIndex && sampleIndex + 1 == rhs.sampleIndex;
}

bool XorChainModel::LoadIndex::gatherable(LoadIndex const& rhs) const {
  return batchMem == rhs.batchMem && batchIndex == rhs.batchIndex && sampleIndex != rhs.sampleIndex;
}

bool XorChainModel::LoadIndex::operator==(LoadIndex const& rhs) const {
  return batchMem == rhs.batchMem && batchIndex == rhs.batchIndex && sampleIndex == rhs.sampleIndex;
}

XorChainModel::XorChain::XorChain(Value value, unsigned lookAhead, BitCodeMap const& bitMap) {
  SmallVector<Value, 2> currentValues{value};
  SmallVector<Value, 8> allOperands;
  for (unsigned remainingLookAhead = lookAhead; remainingLookAhead-- > 0;) {
    SmallVector<Value, 4> operands;
    bool allCommutative = commutative(currentValues.begin(), currentValues.end());
    for (auto currentValue : currentValues) {
      auto currentOperands = getOperands(currentValue);
      if (!allCommutative && commutative(currentValue)) {
        sortByOpcode(currentOperands);
      }
      allOperands.append(currentOperands);
    }
    if (allCommutative) {
      sortByOpcode(operands);
    }
    currentValues.swap(operands);
  }
  unsigned oldSize = sequence.size();
  sequence.resize(oldSize + (allOperands.size() * bitMap.codeWidth()));
  for (size_t i = 0; i < allOperands.size(); ++i) {
    auto const& bitVector = bitMap.lookup(allOperands[i]);
    for (auto it = bitVector.set_bits_begin(); it != bitVector.set_bits_end(); ++it) {
      sequence.set(oldSize + (bitMap.codeWidth() * i) + *it);
    }
    if (auto* definingOp = allOperands[i].getDefiningOp()) {
      if (auto batchRead = dyn_cast<SPNBatchRead>(definingOp)) {
        loads.emplace_back(batchRead.batchMem(), batchRead.batchIndex(), batchRead.sampleIndex());
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
  // Search for consecutive loads, then for gathers, then for broadcasts.
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
  // No match for a load? Downgrade the score.
  // Use 3 since it should be taken into account as worse than consecutive loads, gathers or broadcasts.
  return penalty + (loads.size() - matchedLoads.size() + (rhs.loads.size() - rhsMatchedLoads.size())) * 3;
}

void XorChainModel::XorChain::dump() const {
  dumpBitVector(sequence, "", "\n");
}
