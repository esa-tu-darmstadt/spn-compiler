//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

SLPGraphBuilder::SLPGraphBuilder(SLPGraph& graph) : graph{graph} {}

// Some helper functions in an anonymous namespace.
namespace {
  void computeDepths(ArrayRef<Value> seed, DenseMap<Value, unsigned>& depths) {
    llvm::SmallSetVector<Value, 32> worklist;
    for (auto value : seed) {
      depths[value] = 0;
      worklist.insert(value);
    }
    while (!worklist.empty()) {
      auto value = worklist.pop_back_val();
      if (auto* definingOp = value.getDefiningOp()) {
        for (auto operand: definingOp->getOperands()) {
          auto currentDepth = depths.lookup(operand);
          auto newDepth = depths[value] + 1;
          if (newDepth > currentDepth) {
            depths[operand] = newDepth;
            worklist.insert(operand);
          }
        }
      }
    }
  }
}

void SLPGraphBuilder::build(ArrayRef<Value> seed) {
  graph.superwordRoot = std::make_shared<Superword>(seed);
  graph.nodeRoot = std::make_shared<SLPNode>(graph.superwordRoot);
  nodeBySuperword[graph.superwordRoot.get()] = graph.nodeRoot;
  superwordsByValue[graph.superwordRoot->getElement(0)].emplace_back(graph.superwordRoot);
  if (!option::allowTopologicalMixing) {
    computeDepths(seed, valueDepths);
  }
  buildWorklist.insert(graph.nodeRoot.get());
  buildGraph(graph.superwordRoot);
  //dumpSLPGraph(graph.nodeRoot.get(), true);
}

// Some helper functions in an anonymous namespace.
namespace {

  bool continueBuilding(Superword const& superword, DenseMap<Value, unsigned> const& valueDepths) {
    if (!vectorizable(superword.begin(), superword.end())) {
      return false;
    }
    if (!option::allowDuplicateElements && !allLeaf(superword.begin(), superword.end())) {
      if (SmallPtrSet<Value, 4>{std::begin(superword), std::end(superword)}.size() < superword.numLanes()) {
        return false;
      }
    }
    if (!option::allowTopologicalMixing && !allLeaf(superword.begin(), superword.end())) {
      for (size_t lane = 1; lane < superword.numLanes(); ++lane) {
        if (valueDepths.lookup(superword.getElement(lane)) != valueDepths.lookup(superword.getElement(0))) {
          return false;
        }
      }
    }
    return true;
  }

  bool appendable(SLPNode const& node,
                  OperationName const& opCode,
                  ArrayRef<SmallVector<Value, 2>> allOperands,
                  unsigned operandIndex) {
    if (node.numSuperwords() == option::maxNodeSize) {
      return false;
    }
    return std::all_of(std::begin(allOperands), std::end(allOperands), [&](auto const& operands) {
      auto const& operand = operands[operandIndex];
      if (operand.getDefiningOp()->getName() != opCode) {
        return false;
      }
      // Check if any operand escapes the current node.
      return std::all_of(std::begin(operand.getUsers()), std::end(operand.getUsers()), [&](auto* user) {
        return node.contains(user->getResult(0));
      });
    });
  }

  SmallVector<Value, 2> getOperands(Value value) {
    SmallVector<Value, 2> operands;
    operands.reserve(value.getDefiningOp()->getNumOperands());
    for (auto operand : value.getDefiningOp()->getOperands()) {
      operands.emplace_back(operand);
    }
    return operands;
  }

  void sortByOpcode(SmallVector<Value, 2>& values, Optional<OperationName> const& smallestOpcode) {
    llvm::sort(std::begin(values), std::end(values), [&](Value lhs, Value rhs) {
      auto* lhsOp = lhs.getDefiningOp();
      auto* rhsOp = rhs.getDefiningOp();
      if (!lhsOp && !rhsOp) {
        return lhs.cast<BlockArgument>().getArgNumber() < rhs.cast<BlockArgument>().getArgNumber();
      } else if (lhsOp && !rhsOp) {
        return true;
      } else if (!lhsOp && rhsOp) {
        return false;
      }
      if (smallestOpcode.hasValue()) {
        if (lhsOp->getName() == smallestOpcode.getValue()) {
          return rhsOp->getName() != smallestOpcode.getValue();
        } else if (rhsOp->getName() == smallestOpcode.getValue()) {
          return false;
        }
      }
      if (lhsOp->getName().getStringRef() == rhsOp->getName().getStringRef()) {
        return lhsOp->isBeforeInBlock(rhsOp);
      }
      return lhsOp->getName().getStringRef() < rhsOp->getName().getStringRef();
    });
  }

  SmallVector<SmallVector<Value, 2>> getAllOperandsSorted(Superword const& superword,
                                                          OperationName const& currentOpCode) {
    SmallVector<SmallVector<Value, 2>> allOperands;
    allOperands.reserve(superword.numLanes());
    for (auto value : superword) {
      allOperands.emplace_back(getOperands(value));
    }
    for (auto& operands : allOperands) {
      sortByOpcode(operands, currentOpCode);
    }
    return allOperands;
  }

  struct SuperwordSemantics {
    SuperwordSemantics() = default;
    SuperwordSemantics(Superword* superword, DenseMap<Superword*, SuperwordSemantics> const& parentSemantics)
        : operandDifference{superword->numLanes()} {
      for (unsigned lane = 0; lane < superword->numLanes(); ++lane) {
        auto op = superword->getElement(lane).getDefiningOp();
        for (unsigned i = 0; i < superword->numOperands(); ++i) {
          auto operand = op->getOperand(i);
          auto operandElement = superword->getOperand(i)->getElement(lane);
          if (operand != operandElement) {
            if (operandDifference[lane][operand] == 1) {
              // Keep the map as small as possible.
              operandDifference[lane].erase(operand);
            } else {
              --operandDifference[lane][operand];
            }
            if (operandDifference[lane][operandElement] == -1) {
              // Keep the map as small as possible.
              operandDifference[lane].erase(operandElement);
            } else {
              ++operandDifference[lane][operandElement];
            }
          }
        }
      }
      for (unsigned lane = 0; lane < superword->numLanes(); ++lane) {
        for (auto* operandWord : superword->getOperands()) {
          auto const& semantics = parentSemantics.lookup(operandWord);
          if (semantics.operandDifference.empty()) {
            continue;
          }
          for (auto const& differenceEntry : semantics.operandDifference[lane]) {
            auto newDifference = operandDifference[lane].lookup(differenceEntry.first) + differenceEntry.second;
            if (newDifference == 0) {
              operandDifference[lane].erase(differenceEntry.first);
            } else {
              operandDifference[lane][differenceEntry.first] = newDifference;
            }
          }
        }
      }
    }

    bool areSemanticsAlteredInLane(size_t lane) const {
      return !operandDifference[lane].empty();
    }

    // positive: surplus of that value in the computation chain
    // negative: deficiency of that value in the computation chain
    SmallVector<DenseMap<Value, int>, 4> operandDifference;
  };

} // end namespace

void SLPGraphBuilder::buildGraph(std::shared_ptr<Superword> const& superword) {
  // Stop growing graph
  if (!continueBuilding(*superword, valueDepths)) {
    return;
  }
  auto currentNode = nodeBySuperword[superword.get()];
  auto const& currentOpCode = superword->begin()->getDefiningOp()->getName();
  auto const& arity = superword->begin()->getDefiningOp()->getNumOperands();
  // Recursion call to grow graph further
  // 1. Commutative
  if (commutative(superword->begin(), superword->end())) {
    // A. Coarsening Mode
    auto allOperands = getAllOperandsSorted(*superword, currentOpCode);
    for (unsigned i = 0; i < arity; ++i) {
      SmallVector<Value, 4> superwordValues;
      for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
        superwordValues.emplace_back(allOperands[lane][i]);
      }
      if (auto existingSuperword = superwordOrNull(superwordValues)) {
        superword->addOperand(existingSuperword);
        currentNode->addOperand(nodeBySuperword[existingSuperword.get()]);
      } else if (appendable(*currentNode, currentOpCode, allOperands, i)) {
        auto newSuperword = appendSuperwordToNode(superwordValues, currentNode, superword);
        buildGraph(newSuperword);
      } else if (ofVectorizableType(std::begin(superwordValues), std::end(superwordValues))) {
        auto operandNode = addOperandToNode(superwordValues, currentNode, superword);
        buildWorklist.insert(operandNode.get());
      }
    }
    // B. Normal Mode: Finished building multi-node
    if (currentNode->isSuperwordRoot(*superword)) {
      reorderOperands(currentNode.get());
      // We might want to shuffle superwords later on. We can not shuffle them if their operands have been reordered
      // by more than just a simple commutative swap, since otherwise their semantics would be different.
      // Note: the semantics of the node's root cannot be changed as it accumulates everything, no matter the order.
      if (currentNode->numSuperwords() > 1) {
        DenseMap<Superword*, SuperwordSemantics> semantics;
        for (unsigned i = currentNode->numSuperwords(); i-- > 0;) {
          auto* nodeWord = currentNode->getSuperword(i).get();
          semantics.try_emplace(nodeWord, nodeWord, semantics);
        }
        for (unsigned i = 0; i < currentNode->numSuperwords(); ++i) {
          for (unsigned lane = 0; lane < currentNode->numLanes(); ++lane) {
            if (semantics.lookup(currentNode->getSuperword(i).get()).areSemanticsAlteredInLane(lane)) {
              currentNode->getSuperword(i)->markSemanticsAlteredInLane(lane);
            }
          }
        }
      }
      for (auto const& operandNode : currentNode->getOperands()) {
        if (buildWorklist.erase(operandNode.get())) {
          buildGraph(operandNode->getSuperword(operandNode->numSuperwords() - 1));
        }
      }
    }
  }
    // 2. Non-Commutative
  else {
    for (size_t i = 0; i < arity; ++i) {
      SmallVector<Value, 4> operandValues;
      for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
        auto operand = currentNode->getValue(lane, 0).getDefiningOp()->getOperand(i);
        operandValues.emplace_back(operand);
      }
      if (auto existingSuperword = superwordOrNull(operandValues)) {
        superword->addOperand(existingSuperword);
        currentNode->addOperand(nodeBySuperword[existingSuperword.get()]);
      } else if (ofVectorizableType(std::begin(operandValues), std::end(operandValues))) {
        auto operandNode = addOperandToNode(operandValues, currentNode, superword);
        buildWorklist.insert(operandNode.get());
        buildGraph(operandNode->getSuperword(0));
      }
    }
  }
}

void SLPGraphBuilder::reorderOperands(SLPNode* multinode) const {
  auto const& numOperands = multinode->numOperands();
  assert(numOperands > 0 && "trying to reorder a node with zero operands");
  SmallVector<SmallVector<Value, 4>> finalOrder{multinode->numLanes()};
  SmallVector<SmallVector<Mode, 4>> mode{multinode->numLanes()};
  // 1. Strip first lane
  for (size_t i = 0; i < numOperands; ++i) {
    auto value = multinode->getOperand(i)->getValue(0, 0);
    finalOrder[0].emplace_back(value);
    mode[0].emplace_back(modeFromValue(value));
  }
  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode->numLanes(); ++lane) {
    SmallVector<Value> candidates;
    for (auto const& operand : multinode->getOperands()) {
      candidates.emplace_back(operand->getValue(lane, 0));
    }
    // Look for a matching candidate
    for (size_t i = 0; i < numOperands; ++i) {
      // Skip if we can't vectorize
      if (mode[lane - 1][i] == FAILED) {
        finalOrder[lane].emplace_back(nullptr);
        mode[lane].emplace_back(FAILED);
        continue;
      }
      auto last = finalOrder[lane - 1][i];
      auto const& bestResult = getBest(mode[lane - 1][i], last, candidates);
      // Update output
      finalOrder[lane].emplace_back(bestResult.first);
      // Detect SPLAT mode
      if (i == 1 && bestResult.first == last) {
        mode[lane].emplace_back(SPLAT);
      } else {
        mode[lane].emplace_back(bestResult.second);
      }
    }
    // Distribute remaining candidates in case we encountered a FAILED.
    for (auto candidate : candidates) {
      for (size_t i = 0; i < numOperands; ++i) {
        if (finalOrder[lane][i] == nullptr) {
          finalOrder[lane][i] = candidate;
          break;
        }
      }
    }
  }
  for (size_t operandIndex = 0; operandIndex < multinode->numOperands(); ++operandIndex) {
    for (size_t lane = 0; lane < multinode->numLanes(); ++lane) {
      if (multinode->getOperand(operandIndex)->getValue(lane, 0) != finalOrder[lane][operandIndex]) {
        multinode->getOperand(operandIndex)->setValue(lane, 0, finalOrder[lane][operandIndex]);
      }
    }
  }
}

std::pair<Value, SLPGraphBuilder::Mode> SLPGraphBuilder::getBest(Mode mode,
                                                                 Value last,
                                                                 SmallVector<Value>& candidates) const {
  Value best;
  Mode resultMode = mode;
  SmallVector<Value> bestCandidates;
  if (mode == FAILED) {
    // Don't select now, let others choose first
    best = nullptr;
  } else if (mode == SPLAT) {
    // Look for other splat candidates
    for (auto& operand : candidates) {
      if (operand == last) {
        best = operand;
        break;
      }
    }
  } else {
    // Default value
    best = candidates.front();
    for (auto& candidate : candidates) {
      if (mode == LOAD) {
        if (consecutiveLoads(last, candidate)) {
          bestCandidates.emplace_back(candidate);
        }
      } else if (!last.isa<BlockArgument>() && !candidate.isa<BlockArgument>()) {
        if (last.getDefiningOp()->getName() == candidate.getDefiningOp()->getName()) {
          bestCandidates.emplace_back(candidate);
        }
      }
    }
    // 1. If we have a trivial solution, use it
    // No matches
    if (bestCandidates.empty()) {
      resultMode = FAILED;
    }
      // Single match
    else if (bestCandidates.size() == 1) {
      best = bestCandidates.front();
    }
      // 2. Look-ahead to choose from best candidates
    else if (mode == OPCODE) {
      // Look-ahead on various levels
      // TODO: when the level is increased, we recompute everything from the level before. change that maybe?
      for (size_t level = 1; level <= option::maxLookAhead; ++level) {
        // Best is the candidate with max score
        unsigned bestScore = 0;
        llvm::SmallSet<unsigned, 4> scores;
        for (auto candidate : bestCandidates) {
          // Get the look-ahead score
          unsigned score = getLookAheadScore(last, candidate, level);
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
    }
  }
  // Remove best from candidates
  if (best != nullptr) {
    candidates.erase(std::find(std::begin(candidates), std::end(candidates), best));
  }
  return {best, resultMode};
}

unsigned SLPGraphBuilder::getLookAheadScore(Value last, Value candidate, unsigned maxLevel) const {
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

// === Utilities === //

SLPGraphBuilder::Mode SLPGraphBuilder::modeFromValue(Value value) {
  if (auto* definingOp = value.getDefiningOp()) {
    if (definingOp->hasTrait<OpTrait::ConstantLike>()) {
      return Mode::CONST;
    } else if (dyn_cast<spn::low::SPNBatchRead>(definingOp)) {
      return Mode::LOAD;
    }
    return Mode::OPCODE;
  }
  return Mode::SPLAT;
}

std::shared_ptr<Superword> SLPGraphBuilder::appendSuperwordToNode(ArrayRef<Value> values,
                                                                  std::shared_ptr<SLPNode> const& node,
                                                                  std::shared_ptr<Superword> const& usingSuperword) {
  auto superword = std::make_shared<Superword>(values);
  superwordsByValue[values[0]].emplace_back(superword);
  nodeBySuperword[superword.get()] = node;
  node->addSuperword(superword);
  usingSuperword->addOperand(superword);
  return superword;
}

std::shared_ptr<SLPNode> SLPGraphBuilder::addOperandToNode(ArrayRef<Value> operandValues,
                                                           std::shared_ptr<SLPNode> const& node,
                                                           std::shared_ptr<Superword> const& usingSuperword) {
  auto superword = std::make_shared<Superword>(operandValues);
  superwordsByValue[operandValues[0]].emplace_back(superword);
  auto operandNode = nodeBySuperword.try_emplace(superword.get(), std::make_shared<SLPNode>(superword)).first->second;
  nodeBySuperword[superword.get()] = operandNode;
  node->addOperand(operandNode);
  usingSuperword->addOperand(superword);
  return operandNode;
}

std::shared_ptr<Superword> SLPGraphBuilder::superwordOrNull(ArrayRef<Value> values) const {
  if (superwordsByValue.count(values[0])) {
    auto const& superwords = superwordsByValue.lookup(values[0]);
    auto const& it = std::find_if(std::begin(superwords), std::end(superwords), [&](auto const& superword) {
      if (superword->numLanes() != values.size()) {
        return false;
      }
      for (size_t lane = 1; lane < superword->numLanes(); ++lane) {
        if (superword->hasAlteredSemanticsInLane(lane) || superword->getElement(lane) != values[lane]) {
          return false;
        }
      }
      return true;
    });
    if (it != std::end(superwords)) {
      return *it;
    }
  }
  return nullptr;
}
