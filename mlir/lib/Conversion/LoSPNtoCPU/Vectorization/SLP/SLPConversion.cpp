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

  Value latest(Value const& lhs, Value const& rhs) {
    if (isBeforeInBlock(lhs, rhs)) {
      return rhs;
    }
    return lhs;
  }

  Value latestElement(NodeVector* vector) {
    Value latestElement = vector->getElement(0);
    for (size_t lane = 1; lane < vector->numLanes(); ++lane) {
      latestElement = latest(latestElement, vector->getElement(lane));
    }
    return latestElement;
  }

}

ConversionManager::ConversionManager(SLPNode* root) {
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

Value ConversionManager::getInsertionPoint(NodeVector* vector) const {
  Value latestVal = latestInsertion;
  for (size_t i = 0; i < vector->numOperands(); ++i) {
    auto* operand = vector->getOperand(i);
    if (vectorData.lookup(operand).mode == CreationMode::Skip) {
      continue;
    }
    assert(vectorData.lookup(operand).operation.hasValue() && "operand has not yet been converted");
    auto const& operandValue = vectorData.lookup(operand).operation.getValue();
    if (!latestVal || isBeforeInBlock(latestVal, operandValue)) {
      latestVal = operandValue;
    }
  }
  if (latestVal) {
    return latest(latestVal, latestElement(vector));
  }
  return latestElement(vector);
}

void ConversionManager::update(NodeVector* vector, Value const& operation, CreationMode const& mode) {
  assert(!vectorData[vector].operation.hasValue() && !vectorData[vector].mode.hasValue()
             && "vector has been converted already");
  vectorData[vector].operation = operation;
  vectorData[vector].mode = mode;
  if (!latestInsertion || isBeforeInBlock(latestInsertion, operation)) {
    latestInsertion = operation;
  }
}

void ConversionManager::markSkipped(NodeVector* vector) {
  assert(!vectorData[vector].operation.hasValue() && !vectorData[vector].mode.hasValue()
             && "vector has been converted already");
  vectorData[vector].mode = CreationMode::Skip;
}

bool ConversionManager::isConverted(NodeVector* vector) const {
  return vectorData.lookup(vector).operation.hasValue();
}

Value ConversionManager::getValue(NodeVector* vector) const {
  assert(isConverted(vector) && "vector has not yet been converted");
  return vectorData.lookup(vector).operation.getValue();
}

CreationMode ConversionManager::getCreationMode(NodeVector* vector) const {
  assert(vectorData.lookup(vector).mode.hasValue() && "vector has not yet been converted");
  return vectorData.lookup(vector).mode.getValue();
}

bool ConversionManager::hasEscapingUsers(Value const& value) const {
  return escapingUsers.count(value) && !escapingUsers.lookup(value).empty();
}

Operation* ConversionManager::moveEscapingUsersBehind(NodeVector* vector, Value const& value) const {
  SmallVector<Operation*> users;
  Operation* earliestEscapingUser = nullptr;
  for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
    auto const& element = vector->getElement(lane);
    for (auto* user : escapingUsers.lookup(element)) {
      if (!earliestEscapingUser) {
        earliestEscapingUser = user;
      }
      users.emplace_back(user);
    }
  }
  for (size_t i = 0; i < users.size(); ++i) {
    users.append(std::begin(users[i]->getUsers()), std::end(users[i]->getUsers()));
  }
  std::sort(std::begin(users), std::end(users), [&](Operation* lhs, Operation* rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  users.erase(std::unique(std::begin(users), std::end(users)), std::end(users));

  Operation* latestUser = value.getDefiningOp();
  for (size_t i = 0; i < users.size(); ++i) {
    if (users[i]->isBeforeInBlock(latestUser)) {
      users[i]->moveAfter(latestUser);
    }
    latestUser = users[i];
  }
  return earliestEscapingUser;
}