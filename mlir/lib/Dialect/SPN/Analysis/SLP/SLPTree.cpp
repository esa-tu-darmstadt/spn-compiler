//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPTree.h"
#include "SPN/Analysis/SLP/SLPSeeding.h"

#include <iostream>
#include <algorithm>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPTree::SLPTree(Operation* root, size_t width, size_t maxLookAhead) : graphs{}, maxLookAhead{maxLookAhead} {
  assert(root);
  llvm::StringMap<std::vector<Operation*>> operationsByOpCode;
  for (auto& op : root->getBlock()->getOperations()) {
    operationsByOpCode[op.getName().getStringRef().str()].emplace_back(&op);
    auto const& uses = std::distance(op.getUses().begin(), op.getUses().end());
    if (uses > 1) {
      std::cerr << "SPN is not a tree!" << std::endl;
    }
  }
/*
  auto const& seeds = seeding::getSeeds(root, 4);
  for (auto const& seed : seeds) {
    SLPNode rootNode{seed};
    buildGraph(seed, rootNode);
  }
  std::cout << "seeds computed" << std::endl;
*/
}

void SLPTree::buildGraph(std::vector<Operation*> const& operations, SLPNode& parentNode) {
  for (auto const& op : operations) {
    op->dump();
  }
  // TODO: handle binarizable > multinode conversion: if binarizable node, binarize it before going through here
  // Stop growing graph
  if (!vectorizable(operations)) {
    return;
  }
  // Create new node for values and add to graph
  SLPNode& currentNode = parentNode;
  // Recursion call to grow graph further
  // 1. Commutative
  if (commutative(operations)) {
    // A. Coarsening Mode
    for (auto const& operation : operations) {
      buildGraph(getOperands(operation), currentNode);
    }
    // B. Normal Mode: Finished building multi-node
    if (currentNode.isMultiNode()) {
      reorderOperands(currentNode);
      // TODO buildGraph()
    }
  }
    // 2. Non-Commutative
  else {
    buildGraph(getOperands(operations), currentNode);
  }

}

std::vector<std::vector<SLPNode>> SLPTree::reorderOperands(SLPNode& multinode) {
  assert(multinode.isMultiNode());
  std::vector<std::vector<SLPNode>> finalOrder{multinode.numLanes()};
  std::vector<Mode> mode;
  auto const& numOperands = multinode.getOperands(0).size();
  // 1. Strip first lane
  for (size_t i = 0; i < numOperands; ++i) {
    auto operand = multinode.getOperand(0, i);
    finalOrder.at(0).emplace_back(operand);
    mode.emplace_back(modeFromOperation(operand.getOperation(0, 0)));
  }

  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode.numLanes(); ++lane) {
    std::vector<SLPNode> candidates{multinode.getOperands(lane)};
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

std::pair<Optional<SLPNode>, Mode> SLPTree::getBest(Mode const& mode,
                                                    SLPNode const& last,
                                                    std::vector<SLPNode>& candidates) const {
  Optional<SLPNode> best;
  Mode resultMode = mode;
  std::vector<SLPNode> bestCandidates;

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
      if (candidate.name() == last.name()) {
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
int SLPTree::getLookAheadScore(SLPNode const& last, SLPNode const& candidate, size_t const& maxLevel) const {
  if (maxLevel == 0) {
    // No consecutive loads to check, only opcodes.
    return last.name() == candidate.name() ? 1 : 0;
  }
  auto scoreSum = 0;
  for (size_t lane = 0; lane < last.numLanes(); ++lane) {
    for (auto const& lastOperand : last.getOperands(lane)) {
      for (auto const& candidateOperand : candidate.getOperands(lane)) {
        scoreSum += getLookAheadScore(lastOperand, candidateOperand, maxLevel - 1);
      }
    }
  }
  return scoreSum;
}
