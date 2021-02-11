//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPTree.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPTree::SLPTree(seed_t const& seed, size_t const& maxLookAhead) : operandsOf{}, maxLookAhead{maxLookAhead} {
  auto const& currentNode = std::make_shared<SLPNode>(seed);
  operandsOf[currentNode] = {};
  buildGraph(seed, currentNode);
}

void SLPTree::buildGraph(std::vector<Operation*> const& operations, node_t const& currentNode) {
  llvm::errs() << "Building graph with:\n";
  for (auto* op : operations) {
    llvm::errs() << op << ": ";
    op->dump();
  }
  // Stop growing graph
  if (!vectorizable(operations)) {
    return;
  }
  auto const& currentOpCode = operations.front()->getName();
  // Recursion call to grow graph further
  // 1. Commutative
  if (commutative(operations)) {
    // A. Coarsening Mode
    auto allOperands = getOperands(operations);
    for (auto& operands : allOperands) {
      sortByOpcode(operands, currentOpCode);
    }
    for (size_t i = 0; i < operations.front()->getNumOperands(); ++i) {
      if (std::all_of(std::begin(allOperands), std::end(allOperands), [&](auto const& operandOperations) {
        return operandOperations.at(i)->getName() == currentOpCode && !escapesMultinode(operandOperations.at(i));
      })) {
        for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
          currentNode->addOperationToLane(allOperands.at(lane).at(i), lane);
        }
        buildGraph(currentNode->getLastOperations(), currentNode);
      } else {
        // TODO: here might be a good place to implement variable vector width
        std::vector<Operation*> operandOperations;
        for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
          operandOperations.emplace_back(allOperands.at(lane).at(i));
        }
        for (auto const& tmp : operandOperations) {
          tmp->dump();
        }
        operandsOf[currentNode].emplace_back(std::make_shared<SLPNode>(operandOperations));
      }
    }

    // B. Normal Mode: Finished building multi-node
    if (currentNode->isMultiNode() && currentNode->areRootOfNode(operations)) {
      auto const& order = reorderOperands(currentNode);
      // TODO: reorder based on reordering results
      for (auto& operandNode : operandsOf.at(currentNode)) {
        buildGraph(operandNode->getLastOperations(), currentNode);
      }
    }
  }
    // 2. Non-Commutative
  else {
    std::vector<std::vector<Operation*>> operands;
    for (auto const& operandOperations : getOperandsTransposed(operations)) {
      auto const& operandNode = std::make_shared<SLPNode>(operandOperations);
      operandsOf[currentNode].emplace_back(operandNode);
      buildGraph(operandOperations, operandNode);
    }
  }

}

std::vector<std::vector<Operation*>> SLPTree::reorderOperands(node_t const& multinode) {
  assert(multinode->isMultiNode());
  auto const& numOperands = operandsOf.at(multinode).size();
  std::vector<std::vector<Operation*>> finalOrder{multinode->numLanes()};
  std::vector<std::vector<Mode>> mode{multinode->numLanes()};
  // 1. Strip first lane
  for (size_t i = 0; i < numOperands; ++i) {
    auto operation = operandsOf.at(multinode).at(i)->getOperation(0, 0);
    finalOrder.at(0).emplace_back(operation);
    mode.at(0).emplace_back(modeFromOperation(operation));
  }

  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode->numLanes(); ++lane) {
    std::vector<Operation*> candidates;
    for (auto const& operand : operandsOf.at(multinode)) {
      candidates.emplace_back(operand->getOperation(lane, 0));
    }
    // Look for a matching candidate
    for (size_t i = 0; i < numOperands; ++i) {
      // Skip if we can't vectorize
      // TODO: here might also be a good place to start looking for variable-width
      if (mode.at(lane - 1).at(i) == FAILED) {
        finalOrder.at(lane).emplace_back(nullptr);
        mode.at(lane).emplace_back(FAILED);
        continue;
      }
      auto* last = finalOrder.at(lane - 1).at(i);
      auto const& bestResult = getBest(mode.at(lane - 1).at(i), last, candidates);

      // Update output
      finalOrder.at(lane).emplace_back(bestResult.first);

      // Detect SPLAT mode
      if (i == 1 && bestResult.first == last) {
        mode.at(lane).at(i) = SPLAT;
      } else {
        mode.at(lane).emplace_back(bestResult.second);
      }

    }
    // Distribute remaining candidates in case we encountered a FAILED.
    for (auto* candidate : candidates) {
      for (size_t i = 0; i < numOperands; ++i) {
        if (finalOrder.at(lane).at(i) == nullptr) {
          finalOrder.at(lane).at(i) = candidate;
          break;
        }
      }
    }
  }
  return finalOrder;
}

std::pair<Operation*, Mode> SLPTree::getBest(Mode const& mode,
                                             Operation* last,
                                             std::vector<Operation*>& candidates) const {
  Operation* best = nullptr;
  Mode resultMode = mode;
  std::vector<Operation*> bestCandidates;

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
      // We don't have consecutive memory accesses, therefore we're only interested in opcode comparisons.
      if (candidate->getName() == last->getName()) {
        bestCandidates.emplace_back(candidate);
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
        for (size_t level = 1; level <= maxLookAhead; ++level) {
          // Best is the candidate with max score
          auto bestScore = 0;
          std::set<int> scores;
          for (auto const& candidate : bestCandidates) {
            // Get the look-ahead score
            auto const& score = getLookAheadScore(last, candidate, level);
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
    candidates.erase(std::remove(std::begin(candidates), std::end(candidates), best), std::end(candidates));
  }
  return {best, resultMode};
}

int SLPTree::getLookAheadScore(Operation* last, Operation* candidate, size_t const& maxLevel) const {
  if (maxLevel == 0) {
    // No consecutive loads to check, only opcodes.
    return last->getName() == candidate->getName() ? 1 : 0;
  }
  auto scoreSum = 0;
  // TODO: if operands are commutative, sort operands to speed up lookahead?
  for (auto const& lastOperand : getOperands(last)) {
    for (auto const& candidateOperand : getOperands(candidate)) {
      scoreSum += getLookAheadScore(lastOperand, candidateOperand, maxLevel - 1);

    }
  }
  return scoreSum;
}
