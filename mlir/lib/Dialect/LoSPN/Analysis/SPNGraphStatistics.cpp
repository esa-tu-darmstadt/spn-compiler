//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/Analysis/SPNGraphStatistics.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;

SPNGraphStatistics::SPNGraphStatistics(Operation *root) : rootNode{root} {
  assert(root);
  analyzeGraph(root);
}

void SPNGraphStatistics::analyzeGraph(Operation *root) {
  assert(rootNode);

  llvm::SmallVector<Operation *, 10> ops_inner_add;
  llvm::SmallVector<Operation *, 10> ops_inner_mul;
  llvm::SmallVector<Operation *, 10> ops_leaf_categorical;
  llvm::SmallVector<Operation *, 10> ops_leaf_constant;
  llvm::SmallVector<Operation *, 10> ops_leaf_gaussian;
  llvm::SmallVector<Operation *, 10> ops_leaf_histogram;

  if (auto module = dyn_cast<ModuleOp>(rootNode)) {
    module.walk([&](Operation *op) {
      if (auto leaf = dyn_cast<mlir::spn::low::LeafNodeInterface>(op)) {
        // Handle leaf node.
        if (auto categoricalOp = dyn_cast<SPNCategoricalLeaf>(op)) {
          ops_leaf_categorical.push_back(op);
        } else if (auto gaussianOp = dyn_cast<SPNGaussianLeaf>(op)) {
          ops_leaf_gaussian.push_back(op);
        } else if (auto histOp = dyn_cast<SPNHistogramLeaf>(op)) {
          ops_leaf_histogram.push_back(op);
        } else {
          // Unhandled leaf-node-type: Abort.
          op->emitWarning()
              << "Encountered handled leaf-node-type, operation was a '"
              << op->getName() << "'";
          assert(false);
        }
      } else if (auto constOp = dyn_cast<SPNConstant>(op)) {
        // Handle constant nodes.
        ops_leaf_constant.push_back(op);
      } else {
        // Handle inner node.
        if (auto addOp = dyn_cast<SPNAdd>(op)) {
          ops_inner_add.push_back(op);
        } else if (auto mulOp = dyn_cast<SPNMul>(op)) {
          ops_inner_mul.push_back(op);
        }
      }
    });
  }

  if (!ops_inner_add.empty()) {
    nodeKindCount[ops_inner_add[0]->getName().getStringRef()] =
        ops_inner_add.size();
    innerNodeCount += ops_inner_add.size();
  }
  if (!ops_inner_mul.empty()) {
    nodeKindCount[ops_inner_mul[0]->getName().getStringRef()] =
        ops_inner_mul.size();
    innerNodeCount += ops_inner_mul.size();
  }
  if (!ops_leaf_categorical.empty()) {
    nodeKindCount[ops_leaf_categorical[0]->getName().getStringRef()] =
        ops_leaf_categorical.size();
    leafNodeCount += ops_leaf_categorical.size();
  }
  if (!ops_leaf_constant.empty()) {
    nodeKindCount[ops_leaf_constant[0]->getName().getStringRef()] =
        ops_leaf_constant.size();
    leafNodeCount += ops_leaf_constant.size();
  }
  if (!ops_leaf_gaussian.empty()) {
    nodeKindCount[ops_leaf_gaussian[0]->getName().getStringRef()] =
        ops_leaf_gaussian.size();
    leafNodeCount += ops_leaf_gaussian.size();
  }
  if (!ops_leaf_histogram.empty()) {
    nodeKindCount[ops_leaf_histogram[0]->getName().getStringRef()] =
        ops_leaf_histogram.size();
    leafNodeCount += ops_leaf_histogram.size();
  }
}

unsigned int SPNGraphStatistics::getNodeCount() const {
  return innerNodeCount + leafNodeCount;
}

unsigned int SPNGraphStatistics::getInnerNodeCount() const {
  return innerNodeCount;
}

unsigned int SPNGraphStatistics::getLeafNodeCount() const {
  return leafNodeCount;
}

const llvm::StringMap<unsigned int> &SPNGraphStatistics::getFullResult() const {
  return nodeKindCount;
}

const Operation *SPNGraphStatistics::getRootNode() const { return rootNode; }