//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPTree.h"
#include "SPN/SPNInterfaces.h"

#include <iostream>
#include <algorithm>
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPTree::SLPTree(Operation* root, size_t width) : graph{width} {
  assert(root);
  llvm::StringMap<std::vector<Operation*>> operationsByOpCode;
  for (auto& op : root->getBlock()->getOperations()) {
    operationsByOpCode[op.getName().getStringRef().str()].emplace_back(&op);
  }
  for (auto const& entry : operationsByOpCode) {
    buildGraph(entry.getValue(), graph);
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
  // TODO
  return true;
}

bool SLPTree::commutative(std::vector<Operation*> const& values) const {
  return values.front()->hasTrait<OpTrait::IsCommutative>();
}

bool SLPTree::attachableOperands(OperationName const& currentOperation, OperandRange operands) const {
  // TODO operands escape multi-node?
  return std::all_of(std::begin(operands),
                     std::end(operands),
                     [&](auto const& operand) { return operand.getDefiningOp()->getName() == currentOperation; });
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

void SLPTree::reorderOperands(SLPNode& node) {
  // TODO
}