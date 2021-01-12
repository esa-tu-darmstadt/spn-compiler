//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPTree.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNOpTraits.h"

#include <iostream>
#include <algorithm>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPTree::SLPTree(Operation* root, size_t width) : graphs{} {
  assert(root);
  llvm::StringMap<std::vector<Operation*>> operationsByOpCode;
  for (auto& op : root->getBlock()->getOperations()) {
    operationsByOpCode[op.getName().getStringRef().str()].emplace_back(&op);
  }
  // TODO compute seeds in a proper fashion
  for (auto const& entry : operationsByOpCode) {
    SLPNode rootNode{entry.getValue()};
    buildGraph(entry.getValue(), rootNode);
  }

}

/*
void SLPTree::analyzeGraph(Operation* root) {
  traverseSubgraph(root);
}

void SLPTree::traverseSubgraph(Operation* root) {
  std::cout << root->getName().getStringRef().str() << std::endl;
  if (auto leaf = dyn_cast<LeafNodeInterface>(root)) {
    std::cout << "\tis a leaf." << std::endl;
  } else {
    std::cout << "\tis an inner node." << std::endl;
    for (auto op : root->getOperands()) {
      traverseSubgraph(op.getDefiningOp());
    }
  }

}*/


void SLPTree::buildGraph(std::vector<Operation*> const& values, SLPNode& parentNode) {
  for (auto const& op : values) {
    std::cout << op->getName().getStringRef().str() << std::endl;
  }
  // Stop growing graph
  if (!vectorizable(values)) {
    return;
  }
  // Create new node for values and add to graph
  SLPNode currentNode = parentNode.addOperands(values);
  // Recursion call to grow graph further
  // 1. Commutative
  if (commutative(values)) {
    // A. Coarsening Mode
    for (auto const& operation : values) {
      buildGraph(getOperands(operation), currentNode);
    }
    // B. Normal Mode: Finished building multi-node
    if (currentNode.isMultiNode()) {
      reorderOperands(currentNode);
      // TODO buildGraph() needed for operands? Currently don't think so because of smarter node.addOperands() handling.
    }
  }
    // 2. Non-Commutative
  else {
    buildGraph(getOperands(values), currentNode);
  }

}

bool SLPTree::vectorizable(std::vector<Operation*> const& values) const {
  return std::all_of(std::begin(values), std::end(values), [&](Operation* operation) {
    return operation->hasTrait<OpTrait::spn::Vectorizable>();
  });
}

bool SLPTree::commutative(std::vector<Operation*> const& values) const {
  return std::all_of(std::begin(values), std::end(values), [&](Operation* operation) {
    return operation->hasTrait<OpTrait::IsCommutative>();
  });
}

std::vector<Operation*> SLPTree::getOperands(std::vector<Operation*> const& values) const {
  std::vector<Operation*> operands;
  for (auto const& operation : values) {
    for (auto operand : operation->getOperands()) {
      operands.emplace_back(operand.getDefiningOp());
    }
  }
  return operands;
}

std::vector<Operation*> SLPTree::getOperands(Operation* operation) const {
  std::vector<Operation*> operands;
  for (auto operand : operation->getOperands()) {
    operands.emplace_back(operand.getDefiningOp());
  }
  return operands;
}

SLPTree::MODE SLPTree::modeFromOperation(Operation const* operation) const {
  if (dyn_cast<ConstantOp>(operation)) {
    return MODE::CONST;
  } else if (dyn_cast<HistogramOp>(operation) || dyn_cast<GaussianOp>(operation)) {
    return MODE::LOAD;
  }
  return MODE::OPCODE;
}

std::vector<SLPNode> SLPTree::reorderOperands(SLPNode& multinode) {
  assert(multinode.isMultiNode());
  std::vector<SLPNode> finalOrder;
  std::vector<MODE> mode;
  // 1. Strip first lane
  for (size_t i = 0; i < multinode.getOperands().size(); ++i) {
    auto& operand = multinode.getOperand(i);
    finalOrder.emplace_back(operand);
    mode.emplace_back(modeFromOperation(multinode.getOperation(0, i)));
  }

  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode.numLanes(); ++lane) {
    std::vector<Operation*> candidates = multinode.getLane(lane);
    // Look for a matching candidate
    for (size_t i = 0; i < multinode.getOperands().size(); ++i) {
      // Skip if we can't vectorize
      if (mode.at(i) == MODE::FAILED) {
        continue;
      }
      auto const& last = finalOrder.at(i);
      auto const& bestResult = getBest(mode.at(i), last, candidates);

      // Update output
      // TODO: funky two dimensional stuff
      finalOrder.emplace_back(bestResult.first);

      // Detect SPLAT mode
      if (i == 1 && bestResult.first == last) {
        mode.at(i) = MODE::SPLAT;
      }

    }
  }
  return finalOrder;
}

std::pair<SLPNode, SLPTree::MODE> SLPTree::getBest(SLPTree::MODE const& mode,
                                                   SLPNode const& last,
                                                   std::vector<Operation*> const& candidates) const {
  // TODO
  return {last, mode};
}
