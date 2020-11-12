//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SPNGraphStatistics.h"
#include "SPN/SPNInterfaces.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::spn;

SPNGraphStatistics::SPNGraphStatistics(Operation* root) : rootNode{root} {
  assert(root);
  analyzeGraph(root);
}

void SPNGraphStatistics::analyzeGraph(Operation* root) {
  if (auto query = dyn_cast<QueryInterface>(root)) {
    // If this is a query op, traverse all root nodes stored in the query.
    for (auto r : query.getRootNodes()) {
      analyzeGraph(r);
    }
    if (!nodeKindCount.count(root->getName().getStringRef())) {
      nodeKindCount[root->getName().getStringRef()] = 1;
    } else {
      nodeKindCount[root->getName().getStringRef()] = nodeKindCount[root->getName().getStringRef()] + 1;
    }
  } else {
    traverseSubgraph(root);
  }
}

void SPNGraphStatistics::traverseSubgraph(Operation* root) {
  if (!nodeKindCount.count(root->getName().getStringRef())) {
    nodeKindCount[root->getName().getStringRef()] = 1;
  } else {
    nodeKindCount[root->getName().getStringRef()] = nodeKindCount[root->getName().getStringRef()] + 1;
  }
  if (auto leaf = dyn_cast<LeafNodeInterface>(root)) {
    leafNodeCount++;
  } else {
    innerNodeCount++;
    for (auto op : root->getOperands()) {
      traverseSubgraph(op.getDefiningOp());
    }
  }
}

unsigned int SPNGraphStatistics::getNodeCount() const { return innerNodeCount + leafNodeCount; }

unsigned int SPNGraphStatistics::getInnerNodeCount() const { return innerNodeCount; }

unsigned int SPNGraphStatistics::getLeafNodeCount() const { return leafNodeCount; }

const llvm::StringMap<unsigned int>& SPNGraphStatistics::getFullResult() const { return nodeKindCount; }

const Operation* SPNGraphStatistics::getRootNode() const { return rootNode; }