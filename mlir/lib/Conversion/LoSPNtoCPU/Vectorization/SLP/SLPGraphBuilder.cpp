//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

SLPGraphBuilder::SLPGraphBuilder(size_t maxLookAhead) : maxLookAhead{maxLookAhead} {}

std::shared_ptr<SLPNode> SLPGraphBuilder::build(ArrayRef<Value> const& seed) {
  auto root = std::make_shared<SLPNode>(seed);
  nodes.emplace_back(root);
  reorderWorklist.insert(root.get());
  buildGraph(root->getVector(0), root.get());
  return root;
}

// Some helper functions in an anonymous namespace.
namespace {

  bool appendable(SLPNode* node,
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
        return node->contains(user->getResult(0));
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
    std::sort(std::begin(values), std::end(values), [&](Value const& lhs, Value const& rhs) {
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

  SmallVector<SmallVector<Value, 2>> getAllOperandsSorted(NodeVector* vector, OperationName const& currentOpCode) {
    SmallVector<SmallVector<Value, 2>> allOperands;
    allOperands.reserve(vector->numLanes());
    for (auto const& value : *vector) {
      allOperands.emplace_back(getOperands(value));
    }
    for (auto& operands : allOperands) {
      sortByOpcode(operands, currentOpCode);
    }
    return allOperands;
  }

} // end namespace

void SLPGraphBuilder::buildGraph(NodeVector* vector, SLPNode* currentNode) {
  // Stop growing graph
  if (!vectorizable(vector->begin(), vector->end())) {
    return;
  }
  auto const& currentOpCode = vector->begin()->getDefiningOp()->getName();
  auto const& arity = vector->begin()->getDefiningOp()->getNumOperands();
  // Recursion call to grow graph further
  // 1. Commutative
  if (commutative(vector->begin(), vector->end())) {
    // A. Coarsening Mode
    auto allOperands = getAllOperandsSorted(vector, currentOpCode);
    for (unsigned i = 0; i < arity; ++i) {
      SmallVector<Value, 4> vectorValues;
      for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
        vectorValues.emplace_back(allOperands[lane][i]);
      }
      auto const existingNode = nodeOrNone(vectorValues);
      if (existingNode.hasValue()) {
        currentNode->addOperand(nodes[existingNode->first], existingNode->second, vector);
      } else if (appendable(currentNode, currentOpCode, allOperands, i)) {
        auto* newVector = currentNode->addVector(vectorValues, vector);
        buildGraph(newVector, currentNode);
      } else {
        // TODO: here might be a good place to implement variable vector width
        auto operandNode = std::make_shared<SLPNode>(vectorValues);
        nodes.emplace_back(operandNode);
        reorderWorklist.insert(operandNode.get());
        currentNode->addOperand(operandNode, operandNode->getVector(0), vector);
      }
    }
    // B. Normal Mode: Finished building multi-node
    if (currentNode->isRootOfNode(*vector)) {
      if (reorderWorklist.erase(currentNode)) {
        //reorderOperands(currentNode);
      }
      for (auto* operandNode : currentNode->getOperands()) {
        buildGraph(operandNode->getVector(operandNode->numVectors() - 1), operandNode);
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
      auto const existingNode = nodeOrNone(operandValues);
      if (existingNode.hasValue()) {
        currentNode->addOperand(nodes[existingNode->first], existingNode->second, vector);
      } else {
        auto operandNode = std::make_shared<SLPNode>(operandValues);
        nodes.emplace_back(operandNode);
        reorderWorklist.insert(operandNode.get());
        currentNode->addOperand(operandNode, operandNode->getVector(0), vector);
        buildGraph(operandNode->getVector(0), operandNode.get());
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
        for (size_t level = 1; level <= maxLookAhead; ++level) {
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

SLPGraphBuilder::Mode SLPGraphBuilder::modeFromValue(Value const& value) {
  if (value.isa<BlockArgument>()) {
    return SPLAT;
  }
  auto* definingOp = value.getDefiningOp();
  if (definingOp->hasTrait<OpTrait::ConstantLike>()) {
    return CONST;
  } else if (dyn_cast<mlir::spn::low::SPNBatchRead>(definingOp)) {
    return LOAD;
  }
  return OPCODE;
}

Optional<std::pair<size_t, NodeVector*>> SLPGraphBuilder::nodeOrNone(ArrayRef<Value> const& values) const {
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto const& node = nodes[i];
    if (auto* vector = node->getVectorOrNull(values)) {
      return std::make_pair(i, vector);
    }
  }
  return None;
}
