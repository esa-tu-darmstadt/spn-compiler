//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPNode.h"

#include <iostream>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPNode::SLPNode(size_t const& width) : width{width}, operations{}, operands{} {
}

SLPNode::SLPNode(std::vector<Operation*> const& values) : width{values.size()}, operations{values}, operands{} {
}

SLPNode& SLPNode::addOperands(std::vector<Operation*> const& values) {
  //assert(values.size() == width);
  if (attachable(values)) {
    operations.insert(std::end(operations), std::begin(values), std::end(values));
    return *this;
  }
  operands.emplace_back(SLPNode{values});
  return operands.back();

}

std::vector<Operation*> const& SLPNode::getOperations() {
  return operations;
}

OperationName SLPNode::operationName() {
  return operations.front()->getName();
}

bool SLPNode::isMultiNode() const {
  return (operations.size() / width) > 1;
}

bool SLPNode::attachable(std::vector<Operation*> const& otherOperations) {
  // TODO operands escape multi-node?
  if (operations.empty()) {
    return true;
  }
  return std::all_of(std::begin(otherOperations),
                     std::end(otherOperations),
                     [&](auto const& operation) { return operation->getName() == operationName(); });
}
