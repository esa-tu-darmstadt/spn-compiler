//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SPNNodeLevel.h"
#include "SPN/SPNInterfaces.h"

using namespace mlir::spn;

mlir::spn::SPNNodeLevel::SPNNodeLevel(Operation* root, int rootLevel) : rootNode{root}, rootNodeLevel{rootLevel} {
  assert(root);
  analyzeGraph(root);
}

void SPNNodeLevel::analyzeGraph(Operation* graphRoot) {
  assert(graphRoot);
  if (auto query = dyn_cast<mlir::spn::QueryInterface>(graphRoot)) {
    // If this is a query op, traverse all root nodes stored in the query.
    for (auto root : query.getRootNodes()) {
      analyzeGraph(root);
    }
    opLevels[graphRoot] = 0;
  } else {
    // Simple node, traverse the subgraph rooted at this node.
    traverseSubgraph(graphRoot, {rootNodeLevel});
    auto minmax = std::minmax_element(leafLevels.begin(), leafLevels.end());
    minDepth = *(minmax.first);
    maxDepth = *(minmax.second);
    double pivot = ((double) numLeafNodes) / 2.0;
    double avg_acc = 0;
    int index = 0;
    for (auto it = leafLevels.begin(); it != leafLevels.end(); it++) {
      // Evaluate the histogram to compute average and median.
      avg_acc += *it * leafLevelHistogram[*it];
      index += leafLevelHistogram[*it];
      if (pivot < index) {
        // The median element is located in this bucket.
        if ((numLeafNodes % 2) == 0 && ((int) (pivot + 1)) == index) {
          // Even number of leaf nodes, we need consider two leaves to
          // compute the median depth.
          // Corner case: the second median element is located in the next bucket.
          auto nextBucket = std::next(it);
          medianDepth = ((double) *it + *nextBucket) / 2.0;
        } else {
          medianDepth = *it;
        }
      }
    }
    averageDepth = avg_acc / ((double) numLeafNodes);
  }
}

void SPNNodeLevel::traverseSubgraph(Operation* subgraphRoot, GraphLevelInfo info) {
  assert(subgraphRoot);
  int level = info.level;
  if (opLevels.count(subgraphRoot) && opLevels[subgraphRoot] != info.level) {
    level = std::max(info.level, opLevels[subgraphRoot]);
  }
  opLevels[subgraphRoot] = level;
  if (auto leafNode = dyn_cast<LeafNodeInterface>(subgraphRoot)) {
    // Special treatment of leaf nodes: Stop the recursion
    // and store depth information.

    // Maintain a histogram of the levels of leaf nodes.
    // This will allows us to compute the average & median level later on.
    if (!leafLevels.count(level)) {
      leafLevelHistogram.grow(level);
      leafLevelHistogram[level] = 1;
      leafLevels.insert(level);
    } else {
      leafLevelHistogram[level] = leafLevelHistogram[level] + 1;
    }
    ++numLeafNodes;
  } else {
    // Recurse to child nodes
    for (auto op : subgraphRoot->getOperands()) {
      traverseSubgraph(op.getDefiningOp(), {level + 1});
    }
  }
}

int SPNNodeLevel::getOperationDepth(Operation* op) const {
  if (opLevels.count(op)) {
    return opLevels.lookup(op);
  }
  return -1;
}

int SPNNodeLevel::getMaxDepth() const { return maxDepth; }

int SPNNodeLevel::getMinDepth() const { return minDepth; }

double SPNNodeLevel::getMedianDepth() const { return medianDepth; }

double SPNNodeLevel::getAverageDepth() const { return averageDepth; }

const llvm::DenseMap<mlir::Operation*, int>& SPNNodeLevel::getFullResult() const { return opLevels; }

const mlir::Operation* SPNNodeLevel::getRootNode() const { return rootNode; }