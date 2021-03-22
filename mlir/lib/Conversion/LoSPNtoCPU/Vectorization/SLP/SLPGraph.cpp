//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/Debug.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraph.h"
#include <set>
#include <queue>

using namespace mlir;
using namespace mlir::spn;

SLPGraph::SLPGraph(seed_t const& seed, size_t const& maxLookAhead) : maxLookAhead{maxLookAhead}, root{seed} {
  buildGraph(seed, root);
}

SLPNode& SLPGraph::getRoot() {
  return root;
}

void SLPGraph::dump() const {

  llvm::dbgs() << "digraph debug_graph {\n";
  llvm::dbgs() << "rankdir = BT;\n";
  llvm::dbgs() << "node[shape=box];\n";

  std::queue<SLPNode const*> worklist;
  worklist.emplace(&root);

  while (!worklist.empty()) {
    auto const* node = worklist.front();
    worklist.pop();

    llvm::dbgs() << "node_" << node << "[label=<\n";
    llvm::dbgs() << "\t<TABLE ALIGN=\"CENTER\" BORDER=\"0\" CELLSPACING=\"10\" CELLPADDING=\"0\">\n";
    for (size_t i = node->numVectors(); i-- > 0;) {
      llvm::dbgs() << "\t\t<TR>\n";
      for (size_t lane = 0; lane < node->numLanes(); ++lane) {
        auto* operation = node->getOperation(lane, i);
        llvm::dbgs() << "\t\t\t<TD>";
        llvm::dbgs() << "<B>" << operation->getName() << "</B> <FONT COLOR=\"#bbbbbb\">(" << operation << ")</FONT>";
        llvm::dbgs() << "</TD>";
        if (lane < node->numLanes() - 1) {
          llvm::dbgs() << "<VR/>";
        }
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "\t\t</TR>\n";
    }
    llvm::dbgs() << "\t</TABLE>\n";
    llvm::dbgs() << ">];\n";

    for (auto const& operand : node->getOperands()) {
      llvm::dbgs() << "node_" << node << "->" << "node_" << operand << ";\n";
      worklist.emplace(operand);
    }
  }
  llvm::dbgs() << "}\n";
}

// Some helper functions in an anonymous namespace.
namespace {

  bool areConsecutive(Operation* op1, Operation* op2) {
    auto load1 = dyn_cast<LoadOp>(op1);
    auto load2 = dyn_cast<LoadOp>(op2);
    if (!load1 || !load2) {
      return false;
    }
    if (load1.indices().size() != load2.indices().size()) {
      return false;
    }
    for (size_t i = 0; i < load1.indices().size(); ++i) {
      auto const1 = load1.indices()[i].getDefiningOp<ConstantOp>();
      auto const2 = load2.indices()[i].getDefiningOp<ConstantOp>();
      if (!const1 || !const2) {
        return false;
      }
      auto index1 = const1.value().dyn_cast<IntegerAttr>();
      auto index2 = const2.value().dyn_cast<IntegerAttr>();
      if (!index1 || !index2) {
        return false;
      }
      if (i == load1.indices().size() - 1) {
        return index1.getInt() == index2.getInt() - 1;
      } else if (index1.getInt() != index2.getInt()) {
        return false;
      }
    }
    return false;
  }

  bool vectorizable(std::vector<Operation*> const& operations) {
    for (size_t i = 0; i < operations.size(); ++i) {
      auto* op = operations[i];
      if (!op->hasTrait<OpTrait::OneResult>() || (i > 0 && op->getName() != operations.front()->getName())) {
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

  std::vector<std::vector<Operation*>> getOperandsVectorized(std::vector<Operation*> const& operations) {
    for (auto* operation : operations) {
      assert(operation->getNumOperands() == operations.front()->getNumOperands()
                 && "operations must have same numbers of operands");
    }
    std::vector<std::vector<Operation*>> allOperands;
    for (size_t i = 0; i < operations.front()->getNumOperands(); ++i) {
      std::vector<Operation*> operands;
      operands.reserve(operations.size());
      for (auto* operation : operations) {
        operands.emplace_back(operation->getOperand(i).getDefiningOp());
      }
      allOperands.emplace_back(operands);
    }
    return allOperands;
  }

  void sortByOpcode(std::vector<Operation*>& operations,
                    Optional<OperationName> const& smallestOpcode) {
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

}

void SLPGraph::buildGraph(std::vector<Operation*> const& operations, SLPNode& currentNode) {
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
        for (size_t lane = 0; lane < currentNode.numLanes(); ++lane) {
          currentNode.addOperationToLane(allOperands[lane][i], lane);
        }
        buildGraph(currentNode.getVector(currentNode.numVectors() - 1), currentNode);
      } else {
        // TODO: here might be a good place to implement variable vector width
        std::vector<Operation*> operandOperations;
        for (size_t lane = 0; lane < currentNode.numLanes(); ++lane) {
          operandOperations.emplace_back(allOperands[lane][i]);
        }
        currentNode.addOperand(operandOperations);
      }
    }

    // B. Normal Mode: Finished building multi-node
    if (currentNode.isMultiNode() && currentNode.areRootOfNode(operations)) {
      reorderOperands(currentNode);
      for (auto* operandNode : currentNode.getOperands()) {
        buildGraph(operandNode->getVector(operandNode->numVectors() - 1), *operandNode);
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
      buildGraph(operandVector, currentNode.addOperand(operandVector));
    }
  }
}

void SLPGraph::reorderOperands(SLPNode const& multinode) {
  auto const& numOperands = multinode.numOperands();
  std::vector<std::vector<Operation*>> finalOrder{multinode.numLanes()};
  std::vector<std::vector<Mode>> mode{multinode.numLanes()};
  // 1. Strip first lane
  for (size_t i = 0; i < numOperands; ++i) {
    auto operation = multinode.getOperand(i).getOperation(0, 0);
    finalOrder[0].emplace_back(operation);
    mode[0].emplace_back(modeFromOperation(operation));
  }

  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode.numLanes(); ++lane) {
    std::vector<Operation*> candidates;
    for (auto const& operand : multinode.getOperands()) {
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
  for (size_t operandIndex = 0; operandIndex < multinode.numOperands(); ++operandIndex) {
    for (size_t lane = 0; lane < multinode.numLanes(); ++lane) {
      multinode.getOperand(operandIndex).setOperation(lane, 0, finalOrder[lane][operandIndex]);
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
