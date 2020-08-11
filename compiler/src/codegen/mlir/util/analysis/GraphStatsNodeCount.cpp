//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "GraphStatsNodeCount.h"

using namespace mlir;
using namespace mlir::spn;
using namespace spnc;

GraphStatsNodeCount::GraphStatsNodeCount() : root(nullptr) {}

GraphStatsNodeCount::GraphStatsNodeCount(Operation* _node) : root(_node) {
  update();
}

void GraphStatsNodeCount::update() {
  count_nodes_inner = 0;
  count_nodes_leaf = 0;
  spn_node_counts.clear();

  for (auto nodetype : LISTOF_NODETYPE) {
    spn_node_counts.insert({nodetype, 0});
  }

  visitNode(root);
}

void GraphStatsNodeCount::visitNode(Operation* op) {
  if (op == nullptr) {
    // Encountered nullptr -- abort.
    return;
  }

  bool hasOperands = false;

  if (dyn_cast<SumOp>(op)) {
    hasOperands = true;
    ++(spn_node_counts.find(NODETYPE::SUM)->second);
    ++count_nodes_inner;
  } else if (dyn_cast<ProductOp>(op)) {
    hasOperands = true;
    ++(spn_node_counts.find(NODETYPE::PRODUCT)->second);
    ++count_nodes_inner;
  } else if (dyn_cast<HistogramOp>(op)) {
    ++(spn_node_counts.find(NODETYPE::HISTOGRAM)->second);
    ++count_nodes_leaf;
  } else if (dyn_cast<ConstantOp>(op)) {
    // ToDo: Special handling of constants? Measure for improvement via optimizations?
  } else {
    // Encountered unhandled Op-Type
    SPDLOG_WARN("Unhandled operation type: '{}'", op->getName().getStringRef().str());
  }

  if (hasOperands) {
    for (auto child : op->getOperands()) {
      visitNode(child.getDefiningOp());
    }
  }

}

int GraphStatsNodeCount::getCountNodes(NODETYPE _nodetype) {
  auto nodeCountElement = spn_node_counts.find(_nodetype);
  if (nodeCountElement != spn_node_counts.end()) {
    return nodeCountElement->second;
  } else {
    // Encountered unhandled Op-Type
    SPDLOG_WARN("Unhandled nodetype: '{}'", _nodetype);
    return 0;
  }
}

int GraphStatsNodeCount::getCountNodesInner() const {
  return count_nodes_inner;
}

int GraphStatsNodeCount::getCountNodesLeaf() const {
  return count_nodes_leaf;
}

std::map<NODETYPE, int> GraphStatsNodeCount::getResult() {
  return spn_node_counts;
}

Operation* GraphStatsNodeCount::getRoot() const {
  return root;
}

