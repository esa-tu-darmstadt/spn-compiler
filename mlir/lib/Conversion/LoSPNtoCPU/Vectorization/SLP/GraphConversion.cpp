//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/GraphConversion.h"

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
  for (auto* node : SLPNode::postOrder(*root)) {
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
          if (!operand.isa<BlockArgument>()) {
            auto& users = escapingUsers[operand];
            users.erase(std::remove(std::begin(users), std::end(users), element.getDefiningOp()), std::end(users));
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
    if (vectorData.lookup(operand).flag == ElementFlag::Skip) {
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

void ConversionManager::update(NodeVector* vector, Value const& operation, ElementFlag const& flag) {
  assert(!vectorData[vector].operation.hasValue() && !vectorData[vector].flag.hasValue()
             && "vector has been converted already");
  vectorData[vector].operation = operation;
  vectorData[vector].flag = flag;
  if (!latestInsertion || isBeforeInBlock(latestInsertion, operation)) {
    latestInsertion = operation;
  }
}

void ConversionManager::markSkipped(NodeVector* vector) {
  assert(!vectorData[vector].operation.hasValue() && !vectorData[vector].flag.hasValue()
             && "vector has been converted already");
  vectorData[vector].flag = ElementFlag::Skip;
}

bool ConversionManager::wasConverted(NodeVector* vector) const {
  return vectorData.lookup(vector).operation.hasValue();
}

Value ConversionManager::getValue(NodeVector* vector) const {
  assert(wasConverted(vector) && "vector has not yet been converted");
  return vectorData.lookup(vector).operation.getValue();
}

ElementFlag ConversionManager::getElementFlag(NodeVector* vector) const {
  assert(vectorData.lookup(vector).flag.hasValue() && "vector has not yet been converted");
  return vectorData.lookup(vector).flag.getValue();
}

bool ConversionManager::hasEscapingUsers(Value const& value) const {
  return escapingUsers.count(value) && !escapingUsers.lookup(value).empty();
}

void ConversionManager::recursivelyMoveUsersAfter(NodeVector* vector) const {
  assert(wasConverted(vector) && "vector has not yet been converted");
  Operation* vectorOp = vectorData.lookup(vector).operation->getDefiningOp();
  SmallVector<Operation*> users;
  for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
    auto const& element = vector->getElement(lane);
    for (auto* user : escapingUsers.lookup(element)) {
      if (user->isBeforeInBlock(vectorOp)) {
        users.emplace_back(user);
      }
    }
  }
  for (size_t i = 0; i < users.size(); ++i) {
    for (auto* user : users[i]->getUsers()) {
      if (user->isBeforeInBlock(vectorOp)) {
        users.emplace_back(user);
      }
    }
  }
  std::sort(std::begin(users), std::end(users), [&](Operation* lhs, Operation* rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  users.erase(std::unique(std::begin(users), std::end(users)), std::end(users));

  Operation* latestUser = vectorOp;
  for (size_t i = 0; i < users.size(); ++i) {
    if (users[i]->isBeforeInBlock(latestUser)) {
      users[i]->moveAfter(latestUser);
    }
    latestUser = users[i];
  }
}

Operation* ConversionManager::getEarliestEscapingUser(Value const& value) const {
  assert(hasEscapingUsers(value) && "value does not have any escaping users");
  Operation* earliestUser = nullptr;
  for (auto* user : escapingUsers.lookup(value)) {
    if (!earliestUser) {
      earliestUser = user;
      continue;
    }
    if (user->isBeforeInBlock(earliestUser)) {
      earliestUser = user;
    }
  }
  return earliestUser;
}
