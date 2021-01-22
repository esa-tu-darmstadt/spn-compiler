//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPNode.h"

#include <iostream>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPNode::SLPNode(std::vector<Operation*> const& operations) : operationName{operations.front()->getName()},
                                                              lanes{operations.size()},
                                                              operands{operations.size()} {
  for (size_t i = 0; i < operations.size(); ++i) {
    lanes.at(i).emplace_back(operations.at(i));
  }
}

void SLPNode::addOperands(std::vector<std::vector<Operation*>> const& operandsPerLane) {
  assert(operandsPerLane.size() == numLanes());
  // If the operations are attachable (i.e. same opcode), insert them into this node (forming a multinode).
  if (attachable(operandsPerLane)) {
    for (size_t lane = 0; lane < numLanes(); ++lane) {
      lanes.at(lane).insert(std::end(lanes.at(lane)),
                            std::begin(operandsPerLane.at(lane)),
                            std::end(operandsPerLane.at(lane)));
    }
  }
    // Otherwise, don't modify this node's operations and add the operands as true operands.
  else {
    for (size_t lane = 0; lane < numLanes(); ++lane) {
      addOperandsToLane(operandsPerLane.at(lane), lane);
    }
  }

}

void SLPNode::addOperandsToLane(std::vector<Operation*> const& operations, size_t const& lane) {
  assert(operations.size() == lanes.at(lane).back()->getNumOperands());
  for (auto const& operand : operations) {
    operands.at(lane).emplace_back(SLPNode{{operand}});
  }
}

std::vector<SLPNode>& SLPNode::getOperands(size_t lane) {
  return operands.at(lane);
}

std::vector<Operation*>& SLPNode::getLane(size_t lane) {
  return lanes.at(lane);
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
  return lanes.size();
}

bool SLPNode::attachable(std::vector<std::vector<Operation*>> const& operations) {
  // TODO operands escape multi-node?
  for (auto const& laneOperands : operations) {
    if (std::any_of(std::begin(laneOperands),
                    std::end(laneOperands),
                    [&](auto const& operandOperation) { return operandOperation->getName() != operationName; })) {
      return false;
    }
  }
  return true;
}
