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
      auto const& operand = allOperands.at(i);
      if (std::all_of(std::begin(operand), std::end(operand), [&](auto* operation) {
        return operation->getName() == currentOpCode && !escapesMultinode(operation);
      })) {
        for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
          currentNode->addOperationToLane(operand.at(lane), lane);
        }
        buildGraph(currentNode->getLastOperations(), currentNode);
      } else {
        operandsOf[currentNode].emplace_back(std::make_shared<SLPNode>(operand));
      }
    }

    // B. Normal Mode: Finished building multi-node
    if (currentNode->isMultiNode()) {
      reorderOperands(currentNode);
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

std::vector<std::vector<node_t>> SLPTree::reorderOperands(node_t const& multinode) {
  assert(multinode->isMultiNode());
  std::vector<std::vector<node_t>> finalOrder{multinode->numLanes()};
  std::vector<Mode> mode;
  auto const& numOperands = operandsOf.at(multinode).size();
  // 1. Strip first lane
  for (size_t i = 0; i < numOperands; ++i) {
    auto operand = operandsOf.at(multinode).at(i);
    finalOrder.at(0).emplace_back(operand);
    mode.emplace_back(modeFromOperation(operand->getOperation(0, 0)));
  }

  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode->numLanes(); ++lane) {
    std::vector<node_t> candidates{operandsOf.at(multinode)};
    // Look for a matching candidate
    for (size_t i = 0; i < numOperands; ++i) {
      // Skip if we can't vectorize
      if (mode.at(i) == FAILED) {
        continue;
      }
      auto const& last = finalOrder.at(lane - 1).at(i);
      auto const& bestResult = getBest(mode.at(i), last, candidates);

      // Update output
      finalOrder.at(lane).emplace_back(bestResult.first.getValue());

      // Detect SPLAT mode
      if (i == 1 && bestResult.first == last) {
        mode.at(i) = SPLAT;
      } else {
        mode.at(i) = bestResult.second;
      }

    }
  }
  return finalOrder;
}

std::pair<Optional<node_t>, Mode> SLPTree::getBest(Mode const& mode,
                                                   node_t const& last,
                                                   std::vector<node_t>& candidates) const {
  Optional<node_t> best;
  Mode resultMode = mode;
  std::vector<node_t> bestCandidates;

  if (mode == FAILED) {
    // Don't select now, let others choose first
    best.reset();
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
      //if (candidate.name() == last.name()) {
      bestCandidates.emplace_back(candidate);
      //}
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
          if (best.hasValue() && scores.size() > 1) {
            break;
          }
        }
      }
    }

  }
  // Remove best from candidates
  if (best.hasValue()) {
    candidates.erase(std::remove(std::begin(candidates), std::end(candidates), best), std::end(candidates));
  }
  return {best, resultMode};
}

int SLPTree::getLookAheadScore(node_t const& last, node_t const& candidate, size_t const& maxLevel) const {
  if (maxLevel == 0) {
    // No consecutive loads to check, only opcodes.
    return /*last.name() == candidate.name() ? 1 :*/ 0;
  }
  auto scoreSum = 0;
  for (size_t lane = 0; lane < last->numLanes(); ++lane) {
    for (auto const& lastOperand : operandsOf.at(last)) {
      for (auto const& candidateOperand : operandsOf.at(candidate)) {
        scoreSum += getLookAheadScore(lastOperand, candidateOperand, maxLevel - 1);
      }
    }
  }
  return scoreSum;
}
