//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPTree.h"
#include "SPN/SPNInterfaces.h"

#include <iostream>
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
    std::vector<Operation*> operands;
    for (auto const& operation : values) {
      for (auto operand : operation->getOperands()) {
        operands.emplace_back(operand.getDefiningOp());
      }
    }
    if (attachableOperands(values.front()->getName(), operands)) {
      currentNode.addOperands(operands);
    } else {
      buildGraph(operands, currentNode);
    }

  }

}

bool SLPTree::vectorizable(std::vector<Operation*> const& values) const {
  // TODO
  return true;
}

bool SLPTree::commutative(std::vector<Operation*> const& values) const {
  // TODO
  return true;
}

bool SLPTree::attachableOperands(OperationName const& currentOperation, std::vector<Operation*> const& operands) const {
  // TODO operands escape multi-node?
  for (auto const& operand : operands) {
    if (operand->getName() != currentOperation) {
      return false;
    }
  }
  return true;
}