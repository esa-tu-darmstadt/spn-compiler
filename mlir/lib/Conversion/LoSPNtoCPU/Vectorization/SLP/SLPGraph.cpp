//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPGraph.h"
#include <set>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPGraph::SLPGraph(seed_t const& seed, size_t const& maxLookAhead) : operandsOf{}, maxLookAhead{maxLookAhead} {
  auto const& currentNode = std::make_shared<SLPNode>(seed);
  operandsOf[currentNode] = {};
  buildGraph(seed, currentNode);
  print(*this);
}

std::vector<std::shared_ptr<SLPNode>> SLPGraph::getNodes() const {
  std::vector<std::shared_ptr<SLPNode>> nodes;
  for (auto const& entry : operandsOf) {
    nodes.emplace_back(entry.first);
    for (auto const& operand : entry.second) {
      if (std::find(std::begin(nodes), std::end(nodes), operand) == std::end(nodes)) {
        nodes.emplace_back(operand);
      }
    }
  }
  return nodes;
}

void SLPGraph::buildGraph(std::vector<Operation*> const& operations, node_t const& currentNode) {
  if (std::intptr_t(operations.front()) == 0x5ebd28) {
    llvm::dbgs();
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
        return operandOperations[i]->getName() == currentOpCode && !escapesMultinode(operandOperations[i]);
      })) {
        for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
          currentNode->addOperationToLane(allOperands[lane][i], lane);
        }
        buildGraph(currentNode->getVector(currentNode->numVectors() - 1), currentNode);
      } else {
        // TODO: here might be a good place to implement variable vector width
        std::vector<Operation*> operandOperations;
        for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
          operandOperations.emplace_back(allOperands[lane][i]);
        }
        auto const& operandNode = std::make_shared<SLPNode>(operandOperations);
        operandsOf[currentNode].emplace_back(operandNode);
        operandsOf[operandNode] = {};
      }
    }

    // B. Normal Mode: Finished building multi-node
    if (currentNode->isMultiNode() && currentNode->areRootOfNode(operations)) {
      reorderOperands(currentNode);
      for (auto& operandNode : operandsOf.at(currentNode)) {
        buildGraph(operandNode->getVector(operandNode->numVectors() - 1), operandNode);
      }
    }
  }
    // 2. Non-Commutative
  else {
    // Only consider operands further when the current operations aren't leaf nodes.
    for (auto* operation : operations) {
      for (auto const& operand : operation->getOperands()) {
        if (operand.isa<BlockArgument>()) {
          return;
        }
      }
    }
    for (auto const& operandVector : getOperandsVectorized(operations)) {
      auto const& operandNode = std::make_shared<SLPNode>(operandVector);
      operandsOf[currentNode].emplace_back(operandNode);
      operandsOf[operandNode] = {};
      buildGraph(operandVector, operandNode);
    }
  }
}

void SLPGraph::reorderOperands(node_t const& multinode) {
  assert(multinode->isMultiNode());
  auto const& numOperands = operandsOf.at(multinode).size();
  std::vector<std::vector<Operation*>> finalOrder{multinode->numLanes()};
  std::vector<std::vector<Mode>> mode{multinode->numLanes()};
  // 1. Strip first lane
  for (size_t i = 0; i < numOperands; ++i) {
    auto operation = operandsOf.at(multinode)[i]->getOperation(0, 0);
    finalOrder[0].emplace_back(operation);
    mode[0].emplace_back(modeFromOperation(operation));
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
      if (mode[lane - 1][i] == FAILED) {
        finalOrder[lane].emplace_back(nullptr);
        mode[lane].emplace_back(FAILED);
        continue;
      }
      auto* last = finalOrder[lane - 1][i];
      auto const& bestResult = getBest(mode[lane - 1][i], last, candidates);

      // Update output
      finalOrder[lane].emplace_back(bestResult.first);

      // Detect SPLAT mode
      if (i == 1 && bestResult.first == last) {
        mode[lane][i] = SPLAT;
      } else {
        mode[lane].emplace_back(bestResult.second);
      }

    }
    // Distribute remaining candidates in case we encountered a FAILED.
    for (auto* candidate : candidates) {
      for (size_t i = 0; i < numOperands; ++i) {
        if (finalOrder[lane][i] == nullptr) {
          finalOrder[lane][i] = candidate;
          break;
        }
      }
    }
  }
  for (size_t operandIndex = 0; operandIndex < operandsOf.at(multinode).size(); ++operandIndex) {
    for (size_t lane = 0; lane < multinode->numLanes(); ++lane) {
      operandsOf.at(multinode)[operandIndex]->setOperation(lane, 0, finalOrder[lane][operandIndex]);
    }
  }
}

std::pair<Operation*, Mode> SLPGraph::getBest(Mode const& mode,
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
      if (mode == LOAD) {
        if (areConsecutive(last, candidate)) {
          bestCandidates.emplace_back(candidate);
        }
      } else if (candidate->getName() == last->getName()) {
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
    candidates.erase(std::find(std::begin(candidates), std::end(candidates), best));
  }
  return {best, resultMode};
}

int SLPGraph::getLookAheadScore(Operation* last, Operation* candidate, size_t const& maxLevel) const {
  if (maxLevel == 0) {
    if (last->getName() != candidate->getName()) {
      return 0;
    }
    if (dyn_cast<LoadOp>(last)) {
      return areConsecutive(last, candidate) ? 1 : 0;
    }
    return 1;
  }
  auto scoreSum = 0;
  for (auto& lastOperand : getOperands(last)) {
    for (auto& candidateOperand : getOperands(candidate)) {
      // Can be null if the operand is a block argument.
      if (!lastOperand || !candidateOperand) {
        scoreSum += getLookAheadScore(last, candidate, 0);
      } else {
        scoreSum += getLookAheadScore(lastOperand, candidateOperand, maxLevel - 1);
      }
    }
  }
  return scoreSum;
}
