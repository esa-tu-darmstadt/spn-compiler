//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPConversion.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {
  bool isBeforeInBlock(Value const& lhs, Value const& rhs) {
    if (lhs.isa<BlockArgument>()) {
      return !rhs.isa<BlockArgument>();
    } else if (rhs.isa<BlockArgument>()) {
      return false;
    }
    return lhs.getDefiningOp()->isBeforeInBlock(rhs.getDefiningOp());
  }

  Value latestElement(NodeVector* vector) {
    Value latestVal = vector->getElement(0);
    for (size_t lane = 1; lane < vector->numLanes(); ++lane) {
      if (isBeforeInBlock(latestVal, vector->getElement(lane))) {
        latestVal = vector->getElement(lane);
      }
    }
    return latestVal;
  }

}

ConversionState::ConversionState(SLPNode* root) {
  for (auto* node : SLPNode::postOrder(root)) {
    for (size_t i = node->numVectors(); i-- > 0;) {
      auto* vector = node->getVector(i);
      for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
        auto const& element = vector->getElement(lane);
        if (element.isa<BlockArgument>()) {
          continue;
        }
        if (!escapingUsers.count(element)) {
          escapingUsers[element].assign(std::begin(element.getUsers()), std::end(element.getUsers()));
        }
        for (auto const& operand : element.getDefiningOp()->getOperands()) {
          if (!operand.isa<BlockArgument>() && escapingUsers.count(operand)) {
            auto& users = escapingUsers[operand];
            users.erase(std::remove(std::begin(users), std::end(users), element.getDefiningOp()));
          }
        }
      }
    }
  }
  for (auto& entry : escapingUsers) {
    std::sort(std::begin(entry.second), std::end(entry.second), [&](Operation* lhs, Operation* rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
  }
}

Value ConversionState::getInsertionPoint(NodeVector* vector) const {
  Value insertionPoint = latestInsertion;
  for (size_t i = 0; i < vector->numOperands(); ++i) {
    auto* operand = vector->getOperand(i);
    if (vectorData.lookup(operand).mode == CreationMode::Skip) {
      continue;
    }
    assert(vectorData.lookup(operand).operation.hasValue() && "operand has not yet been converted");
    auto const& operandValue = vectorData.lookup(operand).operation.getValue();
    if (!insertionPoint || isBeforeInBlock(insertionPoint, operandValue)) {
      insertionPoint = operandValue;
    }
  }
  return insertionPoint ? insertionPoint : latestElement(vector);
}

void ConversionState::update(NodeVector* vector, Value const& operation, CreationMode const& mode) {
  assert(!vectorData[vector].operation.hasValue() && !vectorData[vector].mode.hasValue()
             && "vector has been converted already");
  vectorData[vector].operation = operation;
  vectorData[vector].mode = mode;
  if (!latestInsertion || isBeforeInBlock(latestInsertion, operation)) {
    latestInsertion = operation;
  }
}

void ConversionState::markSkipped(NodeVector* vector) {
  assert(!vectorData[vector].operation.hasValue() && !vectorData[vector].mode.hasValue()
             && "vector has been converted already");
  vectorData[vector].mode = CreationMode::Skip;
}

bool ConversionState::isConverted(NodeVector* vector) const {
  return vectorData.lookup(vector).operation.hasValue();
}

Value ConversionState::getValue(NodeVector* vector) const {
  assert(isConverted(vector) && "vector has not yet been converted");
  return vectorData.lookup(vector).operation.getValue();
}

CreationMode ConversionState::getCreationMode(NodeVector* vector) const {
  assert(vectorData.lookup(vector).mode.hasValue() && "vector has not yet been converted");
  return vectorData.lookup(vector).mode.getValue();
}

bool ConversionState::hasEscapingUsers(Value const& value, SmallVectorImpl<Operation*>& users) const {
  if (!escapingUsers.count(value) || escapingUsers.lookup(value).empty()) {
    return false;
  }
  users.assign(escapingUsers.lookup(value));
  return true;
}
