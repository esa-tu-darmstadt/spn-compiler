//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <cassert>
#include "BinaryTreeTransform.h"

using namespace spnc;

NodeReference BinaryTreeTransform::binarizeTree(const NodeReference rootNode) {
  rootNode->accept(*this, nullptr);
  return updated_nodes.at(rootNode->id());
}

void BinaryTreeTransform::transform(IRGraph& input) {
  auto newRoot = binarizeTree(input.rootNode());
  for (auto i : input.inputs()) {
    transformedGraph.create<InputVar>(i->id(), i->index());
  }
  transformedGraph.setRootNode(newRoot);
}

void BinaryTreeTransform::visitHistogram(Histogram& n, arg_t arg) {
  updated_nodes.emplace(n.id(), &n);
}

void BinaryTreeTransform::visitGauss(Gauss& n, arg_t arg) {
  updated_nodes.emplace(n.id(), &n);
}

void BinaryTreeTransform::visitProduct(Product& n, arg_t arg) {
  std::vector<NodeReference> newChildren;
  for (auto& c : n.multiplicands()) {
    c->accept(*this, nullptr);
    newChildren.push_back(updated_nodes.at(c->id()));
  }
  updated_nodes.emplace(n.id(), splitChildren<Product>(newChildren, n.id()));
}

void BinaryTreeTransform::visitSum(Sum& n, arg_t arg) {
  std::vector<NodeReference> newChildren;
  for (auto& c : n.addends()) {
    c->accept(*this, nullptr);
    newChildren.push_back(updated_nodes.at(c->id()));
  }
  updated_nodes.emplace(n.id(), splitChildren<Sum>(newChildren, n.id()));
}

template<class T>
NodeReference BinaryTreeTransform::splitChildren(const std::vector<NodeReference>& children,
                                                 const std::string& prefix) {
  // Recursively split the operation into a balanced tree by splitting
  // the list of children halfway and uniting the subtrees with a new node.
  if (children.size() == 1) {
    return children[0];
  } else if (children.size() == 2) {
    return transformedGraph.create<T>(prefix, children);
  } else if (children.size() > 2) {
    std::size_t const half = children.size() / 2;
    std::vector<NodeReference> split_left(children.begin(), children.begin() + half);
    std::vector<NodeReference> split_right(children.begin() + half, children.end());
    auto leftChild = splitChildren < T > (split_left, prefix + "l");
    auto rightChild = splitChildren < T > (split_right, prefix + "r");
    return transformedGraph.create<T>(prefix, std::vector<NodeReference>{leftChild, rightChild});
  }
  assert(false);
}

void BinaryTreeTransform::visitWeightedSum(WeightedSum& n, arg_t arg) {
  std::vector<WeightedAddend> newChildren;
  for (auto& c : n.addends()) {
    c.addend->accept(*this, nullptr);
    newChildren.push_back(WeightedAddend{updated_nodes.at(c.addend->id()), c.weight});
  }
  updated_nodes.emplace(n.id(), splitWeightedChildren(newChildren, n.id()));
}

NodeReference BinaryTreeTransform::splitWeightedChildren(const std::vector<WeightedAddend>& children,
                                                         const std::string& prefix) {
  // Recursively split the operation into a balanced tree by splitting
  // the list of children halfway and uniting the subtrees with a new node.
  if (children.size() == 2) {
    // Base case of the recursion, just create a new weighted sum.
    return transformedGraph.create<WeightedSum>(prefix, children);
  } else if (children.size() == 3) {
    // Special case for three childen, creates a weighted sum with weight 1 or the left two children and
    // the associated weight for the right child.
    auto rightChild = WeightedAddend{splitWeightedChildren(std::vector<WeightedAddend>{children[1],
                                                                                       children[2]},
                                                           prefix + "r"), 1.0};
    return transformedGraph.create<WeightedSum>(prefix, std::vector<WeightedAddend>{children[0], rightChild});
  } else if (children.size() > 3) {
    std::size_t const half = children.size() / 2;
    std::vector<WeightedAddend> split_left(children.begin(), children.begin() + half);
    std::vector<WeightedAddend> split_right(children.begin() + half, children.end());
    auto leftChild = splitWeightedChildren(split_left, prefix + "l");
    auto rightChild = splitWeightedChildren(split_right, prefix + "r");
    // The weight 1 would be associated with both subtrees, so just use a simple sum node.
    return transformedGraph.create<Sum>(prefix, std::vector<NodeReference>{leftChild, rightChild});
  }
  assert(false);
}

