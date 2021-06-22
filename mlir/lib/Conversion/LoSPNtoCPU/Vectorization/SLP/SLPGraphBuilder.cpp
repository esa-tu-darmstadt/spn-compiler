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

void SLPGraphBuilder::build(ArrayRef<Value> const& seed) {
  graph.root = std::make_shared<Superword>(seed);
  auto rootNode = nodeBySuperword.try_emplace(graph.root.get(), std::make_shared<SLPNode>(graph.root)).first->second;
  superwordsByValue[graph.root->getElement(0)].emplace_back(graph.root);
  buildWorklist.insert(rootNode.get());
  buildGraph(graph.root);
  dumpSLPGraph(rootNode.get());
}

// Some helper functions in an anonymous namespace.
namespace {

  bool appendable(SLPNode const& node,
                  OperationName const& opCode,
                  ArrayRef<SmallVector<Value, 2>> const& allOperands,
                  unsigned operandIndex) {
    return std::all_of(std::begin(allOperands), std::end(allOperands), [&](auto const& operands) {
      auto const& operand = operands[operandIndex];
      if (operand.getDefiningOp()->getName() != opCode) {
        return false;
      }
      // Check if any operand escapes the current node.
      // TODO: determine if multinodes should stop building if an operation escapes it or if simply disallowing reordering in this lane might be better
      return std::all_of(std::begin(operand.getUsers()), std::end(operand.getUsers()), [&](auto* user) {
        return node.contains(user->getResult(0));
      });
    });
  }

  SmallVector<Value, 2> getOperands(Value const& value) {
    SmallVector<Value, 2> operands;
    operands.reserve(value.getDefiningOp()->getNumOperands());
    for (auto operand : value.getDefiningOp()->getOperands()) {
      operands.emplace_back(operand);
    }
    return operands;
  }

  void sortByOpcode(SmallVector<Value, 2>& values, Optional<OperationName> const& smallestOpcode) {
    llvm::sort(std::begin(values), std::end(values), [&](Value const& lhs, Value const& rhs) {
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
      return lhsOp->getName().getStringRef() < rhsOp->getName().getStringRef();
    });
  }

  SmallVector<SmallVector<Value, 2>> getAllOperandsSorted(Superword const& superword,
                                                          OperationName const& currentOpCode) {
    SmallVector<SmallVector<Value, 2>> allOperands;
    allOperands.reserve(superword.numLanes());
    for (auto const& value : superword) {
      allOperands.emplace_back(getOperands(value));
    }
    for (auto& operands : allOperands) {
      sortByOpcode(operands, currentOpCode);
    }
    return allOperands;
  }

} // end namespace

void SLPGraphBuilder::buildGraph(std::shared_ptr<Superword> const& superword) {
  // Stop growing graph
  if (!vectorizable(superword->begin(), superword->end())) {
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
      auto existingSuperword = superwordOrNull(superwordValues);
      if (existingSuperword) {
        superword->addOperand(existingSuperword);
        currentNode->addOperand(nodeBySuperword[existingSuperword.get()]);
      } else if (appendable(*currentNode, currentOpCode, allOperands, i)) {
        auto newSuperword = appendSuperwordToNode(superwordValues, currentNode, superword);
        buildGraph(newSuperword);
      } else if (ofVectorizableType(std::begin(superwordValues), std::end(superwordValues))) {
        // TODO: here might be a good place to implement variable vector width
        auto operandNode = addOperandToNode(superwordValues, currentNode, superword);
        buildWorklist.insert(operandNode.get());
      }
    }
    // B. Normal Mode: Finished building multi-node
    if (currentNode->isSuperwordRoot(*superword)) {
      //reorderOperands(currentNode.get());
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
      auto existingSuperword = superwordOrNull(operandValues);
      if (existingSuperword) {
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
  //llvm::dbgs() << "Reordering multinode " << multinode << " with " << numOperands << " operands (" << multinode->numVectors() << " vectors)\n";
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
      // TODO: here might also be a good place to start looking for variable-width
      if (mode[lane - 1][i] == FAILED) {
        finalOrder[lane].emplace_back(nullptr);
        mode[lane].emplace_back(FAILED);
        continue;
      }
      auto const& last = finalOrder[lane - 1][i];
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
    for (auto const& candidate : candidates) {
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
      multinode->getOperand(operandIndex)->setValue(lane, 0, finalOrder[lane][operandIndex]);
    }
  }
}

std::pair<Value, SLPGraphBuilder::Mode> SLPGraphBuilder::getBest(Mode const& mode,
                                                                 Value const& last,
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
    else {
      if (mode == OPCODE) {
        // Look-ahead on various levels
        // TODO: when the level is increased, we recompute everything from the level before. change that maybe?
        for (size_t level = 1; level <= graph.lookAhead; ++level) {
          // Best is the candidate with max score
          unsigned bestScore = 0;
          llvm::SmallSet<unsigned, 4> scores;
          for (auto const& candidate : bestCandidates) {
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
  }
  // Remove best from candidates
  if (best != nullptr) {
    candidates.erase(std::find(std::begin(candidates), std::end(candidates), best));
  }
  return {best, resultMode};
}

unsigned SLPGraphBuilder::getLookAheadScore(Value const& last, Value const& candidate, unsigned maxLevel) const {
  if (maxLevel == 0 || last.isa<BlockArgument>() || candidate.isa<BlockArgument>()) {
    if (last == candidate) {
      return 1;
    }
    if (last.getDefiningOp<SPNBatchRead>()) {
      return consecutiveLoads(last, candidate);
    }
    if (!last.isa<BlockArgument>() && !candidate.isa<BlockArgument>()) {
      return last.getDefiningOp()->getName() == candidate.getDefiningOp()->getName();
    }
    return 0;
  }
  auto scoreSum = 0;
  for (auto& lastOperand : getOperands(last)) {
    for (auto& candidateOperand : getOperands(candidate)) {
      scoreSum += getLookAheadScore(lastOperand, candidateOperand, maxLevel - 1);
    }
  }
  return scoreSum;
}

// === Utilities === //

std::shared_ptr<Superword> SLPGraphBuilder::appendSuperwordToNode(ArrayRef<Value> const& values,
                                                                  std::shared_ptr<SLPNode> const& node,
                                                                  std::shared_ptr<Superword> const& usingSuperword) {
  auto superword = std::make_shared<Superword>(values);
  superwordsByValue[values[0]].emplace_back(superword);
  nodeBySuperword[superword.get()] = node;
  node->addSuperword(superword);
  usingSuperword->addOperand(superword);
  return superword;
}

std::shared_ptr<SLPNode> SLPGraphBuilder::addOperandToNode(ArrayRef<Value> const& operandValues,
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

std::shared_ptr<Superword> SLPGraphBuilder::superwordOrNull(ArrayRef<Value> const& values) const {
  if (superwordsByValue.count(values[0])) {
    auto const& superwords = superwordsByValue.lookup(values[0]);
    auto const& it = std::find_if(std::begin(superwords), std::end(superwords), [&](auto const& superword) {
      if (superword->numLanes() != values.size()) {
        return false;
      }
      for (size_t lane = 1; lane < superword->numLanes(); ++lane) {
        if (superword->getElement(lane) != values[lane]) {
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
