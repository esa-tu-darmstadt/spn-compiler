//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "GraphStatsNodeLevel.h"

using namespace mlir;
using namespace mlir::spn;
using namespace spnc;

GraphStatsNodeLevel::GraphStatsNodeLevel() : root(nullptr), root_level(0) {}

GraphStatsNodeLevel::GraphStatsNodeLevel(Operation* _root, int _rootlevel = 0) : root(_root), root_level(_rootlevel) {
  update();
}

void GraphStatsNodeLevel::update() {
  depth_average = 0.0;
  spn_node_stats.clear();

  for (auto nodetype : LISTOF_NODETYPE) {
    spn_node_stats.insert({nodetype, std::multiset<int>()});
  }

  std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({root_level}));

  visitNode(root, passed_arg);
  processResults();
}

void GraphStatsNodeLevel::visitNode(Operation* op, const arg_t& arg) {
  if (op == nullptr) {
    // Encountered nullptr -- abort.
    return;
  }

  bool hasOperands = false;
  int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;

  if (dyn_cast<SumOp>(op)) {
    hasOperands = true;
    spn_node_stats.find(NODETYPE::SUM)->second.insert(currentLevel);
  } else if (dyn_cast<ProductOp>(op)) {
    hasOperands = true;
    spn_node_stats.find(NODETYPE::PRODUCT)->second.insert(currentLevel);
  } else if (dyn_cast<HistogramOp>(op)) {
    spn_node_stats.find(NODETYPE::HISTOGRAM)->second.insert(currentLevel);
  } else if (dyn_cast<ConstantOp>(op)) {
    // ToDo: Special handling of constants? Measure for improvement via optimizations?
  } else {
    // Encountered unhandled Op-Type
    SPDLOG_WARN("Unhandled operation type: '{}'", op->getName().getStringRef().str());
  }

  if (hasOperands) {
    for (auto child : op->getOperands()) {
      arg_t passed_arg(new GraphStatLevelInfo({currentLevel + 1}));
      visitNode(child.getDefiningOp(), passed_arg);
    }
  }

}

void GraphStatsNodeLevel::processResults() {
  int count_node_temp = 0;
  int count_nodes_leaf = 0;

  for (NODETYPE n : LISTOF_NODETYPE_LEAF) {
    auto nodes = spn_node_stats.find(n)->second;
    count_node_temp = nodes.size();

    if (count_node_temp <= 0) {
      continue;
    }

    count_nodes_leaf += count_node_temp;

    // Note: end() will (unlike begin()) point "behind" the data we're looking for, hence the decrement.
    depth_min = *nodes.begin();
    depth_max = *(--nodes.end());

    // Since the used multimap is ordered, we can simply use the respective node count to get the median index.
    int median_index_temp = count_node_temp / 2;

    // ToDo: Determining the median depth has to be revisited once other leaf nodes are supported (2020-JAN-31).
    for (auto& level : nodes) {
      // level = node.first;
      depth_average += level;

      if (median_index_temp > 0) {
        --median_index_temp;
        if (median_index_temp == 0) {
          depth_median = level;
        }
      }
    }

  }

  depth_average = depth_average / count_nodes_leaf;
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

std::map<NODETYPE, std::multiset<int>> GraphStatsNodeLevel::getResult() {
  return spn_node_stats;
}

Operation* GraphStatsNodeLevel::getRoot() const {
  return root;
}
