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

SLPTree::SLPTree(Operation* root) {
  assert(root);
  llvm::StringMap<std::vector<Operation*>> operationsByOpCode;
  for (auto& op : root->getBlock()->getOperations()) {
    operationsByOpCode[op.getName().getStringRef().str()].emplace_back(&op);
  }
  for (auto const& entry : operationsByOpCode) {
    buildGraph(entry.getValue());
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


void SLPTree::buildGraph(std::vector<Operation*> const& values) {
  for (auto const& op : values) {
    std::cout << op->getName().getStringRef().str() << std::endl;
  }
  // Stop growing graph
  if (!vectorizable(values)) {
    return;
  }
  // Create new node for values and add to graph
  graph.emplace_back(values);
  // Recursion call to grow graph further
  // 1. Commutative
  if (commutative(values)) {
    // A. Coarsening Mode
    for (auto const& operation : values) {
      if (attachableOperands(operation)) {
        std::vector<Operation*> operands;
        for (auto operand : operation->getOperands()) {
          operands.emplace_back(operand.getDefiningOp());
        }
        buildGraph(operands);
      } else {

      }

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

bool SLPTree::attachableOperands(Operation* operation) const {
  // TODO operands escape multi-node?
  for (auto const& operand : operation->getOperands()) {
    if (operand.getDefiningOp()->getName() != operation->getName()) {
      return false;
    }
  }
  return true;
}