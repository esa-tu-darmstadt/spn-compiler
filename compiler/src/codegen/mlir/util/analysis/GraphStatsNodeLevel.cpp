//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "GraphStatsNodeLevel.h"
#include <numeric>

using namespace mlir;
using namespace mlir::spn;
using namespace spnc;

GraphStatsNodeLevel::GraphStatsNodeLevel() : root(nullptr), root_level(0) {}

GraphStatsNodeLevel::GraphStatsNodeLevel(Operation* _root, int _rootlevel = 0) : root(_root), root_level(_rootlevel) {
  update();
}

void GraphStatsNodeLevel::update() {
  depth_average = 0.0;
  spn_op_levels.clear();

  std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({root_level}));

  visitNode(root, passed_arg);
  processResults();
}

void GraphStatsNodeLevel::visitNode(Operation* op, const arg_t& arg) {
  if (op == nullptr) {
    // Encountered nullptr -- abort.
    return;
  }

  int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;

  spn_op_levels.emplace(op, currentLevel);
  auto operands = op->getOperands();

  // Operations with more than one operand -- assume: inner node, e.g. sum or product.
  // Operations with one operand -- assume: leaf node.
  // Operations with no operand -- assume: constant.
  // ToDo: Special handling of constants? Measure for improvement via optimizations?
  if (operands.size() > 1) {
    for (auto child : operands) {
      arg_t passed_arg(new GraphStatLevelInfo({currentLevel + 1}));
      visitNode(child.getDefiningOp(), passed_arg);
    }
  } else if (operands.size() == 1) {
    leaf_levels.insert(currentLevel);
  }
}

void GraphStatsNodeLevel::processResults() {
  int leaves = leaf_levels.size();

  if (leaves > 0) {
    // Note: end() will (unlike begin()) point "behind" the data we're looking for, hence the decrement.
    depth_min = *leaf_levels.begin();
    depth_max = *(--leaf_levels.end());

    int sum = std::accumulate(leaf_levels.begin(), leaf_levels.end(), 0);
    depth_average = (double) sum / leaves;

    // Since the used multiset is ordered, we can simply use the respective node count to get the median index.
    int median_index = leaves / 2;

    if ((median_index >= 0) && (median_index < leaves)) {
      auto median_it = leaf_levels.begin();
      std::advance(median_it, median_index);
      depth_median = *median_it;
    }
  }

}

int GraphStatsNodeLevel::getDepthOperation(Operation* op) const {
  int depth = -1;
  auto it = spn_op_levels.find(op);

  if (it != spn_op_levels.end()) {
    depth = it->second;
  }

  return depth;
}

int GraphStatsNodeLevel::getDepthMax() const {
  return depth_max;
}

int GraphStatsNodeLevel::getDepthMin() const {
  return depth_min;
}

int GraphStatsNodeLevel::getDepthMedian() const {
  return depth_median;
}

double GraphStatsNodeLevel::getDepthAvg() const {
  return depth_average;
}

std::map<Operation*, int> GraphStatsNodeLevel::getResult() {
  return spn_op_levels;
}

Operation* GraphStatsNodeLevel::getRoot() const {
  return root;
}
