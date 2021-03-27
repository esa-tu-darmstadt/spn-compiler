//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/Debug.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include <set>
#include <queue>
#include <LoSPN/LoSPNInterfaces.h>

using namespace mlir;
using namespace mlir::spn::low::slp;

SLPGraphBuilder::SLPGraphBuilder(size_t maxLookAhead) : maxLookAhead{maxLookAhead} {}

std::unique_ptr<SLPNode> SLPGraphBuilder::build(seed_t const& seed) const {
  auto root = std::make_unique<SLPNode>(SLPNode{seed});
  buildGraph(seed, root.get());
  return root;
}

// Some helper functions in an anonymous namespace.
namespace {

  bool vectorizable(std::vector<Operation*> const& operations) {
    for (size_t i = 0; i < operations.size(); ++i) {
      auto* op = operations[i];
      if (!op->hasTrait<OpTrait::spn::low::VectorizableOp>() || (op->getName() != operations.front()->getName())) {
        return false;
      }
    }
    return true;
  }

  bool commutative(std::vector<Operation*> const& operations) {
    return std::all_of(std::begin(operations), std::end(operations), [&](Operation* op) {
      if (op->hasTrait<OpTrait::IsCommutative>()) {
        return true;
      }
      return dyn_cast<AddFOp>(op) || dyn_cast<MulFOp>(op);
    });
  }

  bool escapesMultinode(Operation* operation) {
    // TODO: check if some intermediate, temporary value of a multinode is used outside of it
    assert(operation);
    return false;
  }

  std::vector<Operation*> getOperands(Operation* operation) {
    std::vector<Operation*> operands;
    operands.reserve(operation->getNumOperands());
    for (auto operand : operation->getOperands()) {
      operands.emplace_back(operand.getDefiningOp());
    }
    return operands;
  }

  std::vector<std::vector<Operation*>> getOperands(std::vector<Operation*> const& operations) {
    std::vector<std::vector<Operation*>> allOperands;
    allOperands.reserve(operations.size());
    for (auto* operation : operations) {
      allOperands.emplace_back(getOperands(operation));
    }
    return allOperands;
  }

  void sortByOpcode(std::vector<Operation*>& operations, Optional<OperationName> const& smallestOpcode) {
    std::sort(std::begin(operations), std::end(operations), [&](Operation* lhs, Operation* rhs) {
      if (smallestOpcode.hasValue()) {
        if (lhs->getName() == smallestOpcode.getValue()) {
          return rhs->getName() != smallestOpcode.getValue();
        } else if (rhs->getName() == smallestOpcode.getValue()) {
          return false;
        }
      }
      return lhs->getName().getStringRef() < rhs->getName().getStringRef();
    });
  }
} // end namespace

void SLPGraphBuilder::buildGraph(std::vector<Operation*> const& operations,
                                 SLPNode* currentNode) const {
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
        return operandOperations[i]->getName() == currentOpCode
            && !escapesMultinode(operandOperations[i]);
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
        currentNode->addOperand(operandOperations);
      }
    }
    // B. Normal Mode: Finished building multi-node
    if (currentNode->isMultiNode() && currentNode->areRootOfNode(operations)) {
      reorderOperands(currentNode);
      for (auto* operandNode : currentNode->getOperands()) {
        buildGraph(operandNode->getVector(operandNode->numVectors() - 1), operandNode);
      }
    }
  }
    // 2. Non-Commutative
  else {
    for (size_t i = 0; i < operations.front()->getNumOperands(); ++i) {
      std::vector<Value> operands;
      bool containsBlockArgument = false;
      for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
        auto operand = currentNode->getOperation(lane, 0)->getOperand(i);
        operands.emplace_back(operand);
        containsBlockArgument |= operand.isa<BlockArgument>();
      }
      // Only consider operands that don't contain block arguments for further graph building.
      if (containsBlockArgument) {
        for (auto const& operand : operands) {
          currentNode->addNodeInput(operand);
        }
      } else {
        std::vector<Operation*> operandOperations;
        operandOperations.reserve(operands.size());
        for (auto const& operand : operands) {
          operandOperations.emplace_back(operand.getDefiningOp());
        }
        buildGraph(operandOperations, currentNode->addOperand(operandOperations));
      }
    }
  }
}

void SLPGraphBuilder::reorderOperands(SLPNode* multinode) const {
  if (std::uintptr_t(multinode->getOperation(0, 0)) == 12345) {
    multinode->dump();
    llvm::dbgs() << "\n";
  }
  auto const& numOperands = multinode->numOperands();
  std::vector<std::vector<Operation*>> finalOrder{multinode->numLanes()};
  std::vector<std::vector<Mode>> mode{multinode->numLanes()};
  // 1. Strip first lane
  for (size_t i = 0; i < numOperands; ++i) {
    auto operation = multinode->getOperand(i)->getOperation(0, 0);
    finalOrder[0].emplace_back(operation);
    mode[0].emplace_back(modeFromOperation(operation));
  }
  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode->numLanes(); ++lane) {
    std::vector<Operation*> candidates;
    for (auto const& operand : multinode->getOperands()) {
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
  for (size_t operandIndex = 0; operandIndex < multinode->numOperands(); ++operandIndex) {
    for (size_t lane = 0; lane < multinode->numLanes(); ++lane) {
      multinode->getOperand(operandIndex)->setOperation(lane, 0, finalOrder[lane][operandIndex]);
    }
  }
}

std::pair<Operation*, Mode> SLPGraphBuilder::getBest(Mode const& mode,
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
        if (areConsecutiveLoads(last, candidate)) {
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

int SLPGraphBuilder::getLookAheadScore(Operation* last, Operation* candidate, size_t const& maxLevel) const {
  if (maxLevel == 0) {
    if (last->getName() != candidate->getName()) {
      return 0;
    }
    if (dyn_cast<LoadOp>(last)) {
      return areConsecutiveLoads(last, candidate) ? 1 : 0;
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
