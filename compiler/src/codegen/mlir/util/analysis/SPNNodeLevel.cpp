//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNNodeLevel.h"
#include <numeric>

using namespace mlir;
using namespace mlir::spn;
using namespace spnc;

SPNNodeLevel::SPNNodeLevel() : root(nullptr), root_level(0) {}

SPNNodeLevel::SPNNodeLevel(Operation* _root, int _rootlevel = 0) : root(_root), root_level(_rootlevel) {
  update();
}

void SPNNodeLevel::update() {
  depth_max = 0;
  depth_min = std::numeric_limits<int>::max();
  depth_average = 0.0;
  depth_median = 0.0;
  spn_op_levels.clear();
  leaf_levels.clear();

  std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({root_level}));

  visitNode(root, passed_arg);
  processResults();
}

void SPNNodeLevel::visitNode(Operation* op, const arg_t& arg) {
  if (op == nullptr) {
    // Encountered nullptr -- abort.
    return;
  }

  int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;

  spn_op_levels.emplace(op, currentLevel);
  auto operands = op->getOperands();

  // Operations with more than one operand -- possible: inner node, e.g. sum or product.
  // Operations with one operand -- possible: leaf node.
  // Operations with no operand -- possible: constant.
  // ToDo: Special handling of constants? Measure for improvement via optimizations?
  if (operands.size() > 1) {
    for (auto child : operands) {
      arg_t passed_arg(new GraphStatLevelInfo({currentLevel + 1}));
      visitNode(child.getDefiningOp(), passed_arg);
    }
  } else if (operands.size() == 1) {
    // Add level of an encountered leaf.
    // NOTE: ATM there is only one leaf type, others will have to be added with a dyn_cast<> as well.
    if (dyn_cast<HistogramOp>(op)) {
      leaf_levels.insert(currentLevel);
    }
  }
}

void SPNNodeLevel::processResults() {
  int leaves = leaf_levels.size();

  if (leaves > 0) {
    // Note: end() will (unlike begin()) point "behind" the data we're looking for, hence the decrement.
    depth_min = *leaf_levels.begin();
    depth_max = *(--leaf_levels.end());

    int sum = std::accumulate(leaf_levels.begin(), leaf_levels.end(), 0);
    depth_average = (double) sum / leaves;

    // Since the used multiset is ordered, we can simply use the respective node count to get the (lower) median index.
    int median_index = leaves / 2;
    auto median_it = leaf_levels.begin();

    // Advance iterator to (one of) the median element(s).
    if ((median_index > 0) && (median_index < leaves)) {
      std::advance(median_it, median_index);
    }

    // "odd" case: No further actions required.
    depth_median = *median_it;

    // "even" case: add the second median element and store the average.
    if ((leaves % 2) == 0) {
      if ((median_index + 1) < leaves) {
        depth_median += *(++median_it);
      } else {
        // Corner case -- two leaves -> use previous(!) index.
        depth_median += *(--median_it);
      }
      depth_median = (double) depth_median / 2;
    }
  }

}

int SPNNodeLevel::getDepthOperation(Operation* op) const {
  int depth = -1;
  auto it = spn_op_levels.find(op);

  if (it != spn_op_levels.end()) {
    depth = it->second;
  }

  return depth;
}

int SPNNodeLevel::getDepthMax() const {
  return depth_max;
}

int SPNNodeLevel::getDepthMin() const {
  return depth_min;
}

double SPNNodeLevel::getDepthMedian() const {
  return depth_median;
}

double SPNNodeLevel::getDepthAvg() const {
  return depth_average;
}

std::map<Operation*, int> SPNNodeLevel::getResult() {
  return spn_op_levels;
}

Operation* SPNNodeLevel::getRoot() const {
  return root;
}
