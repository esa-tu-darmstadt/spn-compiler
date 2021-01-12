//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPNode.h"

#include <iostream>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPNode::SLPNode(std::vector<Operation*> const& values) : width{values.size()},
                                                          operationName{values.front()->getName()}, lanes{},
                                                          operands{} {
  lanes.emplace_back(values);
}

SLPNode& SLPNode::addOperands(std::vector<Operation*> const& values) {
  // If the operations are attachable (i.e. same opcode), insert them into this node (forming a multinode).
  if (attachable(values)) {
    for (size_t lane = 0; lane < values.size(); ++lane) {
      lanes.at(lane).emplace_back(values.at(lane));
    }
    return *this;
  }
  operands.emplace_back(SLPNode{values});
  return operands.back();

}

std::vector<SLPNode>& SLPNode::getOperands() {
  return operands;
}

SLPNode& SLPNode::getOperand(size_t index) {
  return operands.at(index);
}

std::vector<Operation*> SLPNode::getLane(size_t laneIndex) {
  std::vector<Operation*> lane;
  for (auto const& operations : lanes) {
    lane.emplace_back(operations.at(laneIndex));
  }
  return lane;
}

Operation* SLPNode::getOperation(size_t lane, size_t index) {
  return lanes.at(lane).at(index);
}

OperationName const& SLPNode::name() {
  return operationName;
}

bool SLPNode::isMultiNode() const {
  return lanes.front().size() > 1;
}

size_t SLPNode::numLanes() const {
  return width;
}

bool SLPNode::attachable(std::vector<Operation*> const& otherOperations) {
  // TODO operands escape multi-node?
  assert(lanes.size() == otherOperations.size());
  return std::all_of(std::begin(otherOperations),
                     std::end(otherOperations),
                     [&](auto const& operation) { return operation->getName() == operationName; });
}
