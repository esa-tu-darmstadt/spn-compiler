//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPNode.h"

#include <iostream>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPNode::SLPNode(std::vector<Operation*> const& operations) : lanes{operations.size()} {
  for (size_t i = 0; i < operations.size(); ++i) {
    lanes.at(i).emplace_back(operations.at(i));
  }
}

void SLPNode::addOperationToLane(Operation* operation, size_t const& lane) {
  lanes.at(lane).emplace_back(operation);
}

std::vector<Operation*> SLPNode::getLastOperations() const {
  std::vector<Operation*> lastOperations;
  for (size_t lane = 0; lane < numLanes(); ++lane) {
    lastOperations.emplace_back(lanes.at(lane).back());
  }
  return lastOperations;
}

Operation* SLPNode::getOperation(size_t lane, size_t index) {
  return lanes.at(lane).at(index);
}

bool SLPNode::isMultiNode() const {
  return lanes.front().size() > 1;
}

bool SLPNode::areRootOfNode(std::vector<Operation*> const& operations) const {
  for (size_t lane = 0; lane < numLanes(); ++lane) {
    if (operations.at(lane) != lanes.at(lane).front()) {
      return false;
    }
  }
  return true;
}

size_t SLPNode::numLanes() const {
  return lanes.size();
}
