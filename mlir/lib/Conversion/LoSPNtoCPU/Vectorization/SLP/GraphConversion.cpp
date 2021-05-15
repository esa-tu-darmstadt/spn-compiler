//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/GraphConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {

  bool later(Value const& lhs, Value const& rhs) {
    if (lhs.isa<BlockArgument>()) {
      return false;
    } else if (rhs.isa<BlockArgument>()) {
      return true;
    }
    return rhs.getDefiningOp()->isBeforeInBlock(lhs.getDefiningOp());
  }

  Value latestElement(ValueVector* vector) {
    Value latestElement = nullptr;
    for (auto const& value : *vector) {
      if (!latestElement || later(value, latestElement)) {
        latestElement = value;
      }
    }
    return latestElement;
  }

  bool willBeFolded(ValueVector* vector) {
    for (auto const& element : *vector) {
      if (auto* definingOp = element.getDefiningOp()) {
        if (!definingOp->hasTrait<OpTrait::ConstantLike>()) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  }

  void reorderOperations(Operation* firstInput,
                         SmallPtrSetImpl<Operation*> const& inputs,
                         SmallPtrSetImpl<Operation*> const& users) {
    DenseMap<Operation*, unsigned> depths;
    SmallVector<Operation*> worklist;
    for (auto* user : users) {
      worklist.emplace_back(user);
      depths[user] = 0;
    }
    unsigned maxDepth = 0;
    while (!worklist.empty()) {
      auto* currentOp = worklist.pop_back_val();
      for (auto const& operand : currentOp->getOperands()) {
        if (auto* operandOp = operand.getDefiningOp()) {
          if (operandOp->isBeforeInBlock(firstInput)) {
            continue;
          }
          unsigned operandDepth = depths[currentOp] + 1;
          if (operandDepth > depths[operandOp]) {
            depths[operandOp] = operandDepth;
            maxDepth = std::max(maxDepth, operandDepth);
            worklist.emplace_back(operandOp);
          }
        }
      }
    }
    // Sort operations in between the first input & the latest escaping user.
    SmallVector<SmallVector<Operation*>> opsSortedByDepth{maxDepth + 1};
    for (auto const& entry : depths) {
      opsSortedByDepth[maxDepth - entry.second].emplace_back(entry.first);
    }
    for (auto& ops: opsSortedByDepth) {
      std::sort(std::begin(ops), std::end(ops), [&](Operation* lhs, Operation* rhs) {
        return lhs->isBeforeInBlock(rhs);
      });
    }
    Operation* latestOp = firstInput;
    for (unsigned depth = 0; depth <= maxDepth; ++depth) {
      auto const& ops = opsSortedByDepth[depth];
      for (auto* op : ops) {
        op->moveAfter(latestOp);
        latestOp = op;
      }
    }
  }

}

ConversionManager::ConversionManager(SLPNode* root, PatternRewriter& rewriter) : order{
    graph::postOrder(root->getVector(0))}, rewriter{rewriter}, folder{root->getValue(0, 0).getContext()} {
  Operation* earliestInput = nullptr;
  SmallPtrSet<Operation*, 32> inputs;
  for (auto const* vector : order) {
    for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
      auto const& element = vector->getElement(lane);
      if (auto* elementOp = element.getDefiningOp()) {
        if (!escapingUsers.count(element)) {
          escapingUsers[element].assign(std::begin(element.getUsers()), std::end(element.getUsers()));
        }
        for (auto const& operand : elementOp->getOperands()) {
          auto& users = escapingUsers[operand];
          users.erase(std::remove(std::begin(users), std::end(users), elementOp), std::end(users));
        }
        if (vector->isLeaf()) {
          if (!earliestInput || elementOp->isBeforeInBlock(earliestInput)) {
            earliestInput = elementOp;
          }
          inputs.insert(elementOp);
        }
      }
    }
  }
  if (earliestInput) {
    Operation* latestEscapingUser = nullptr;
    SmallPtrSet<Operation*, 32> users;
    for (auto& entry : escapingUsers) {
      for (auto* escapingUser : entry.second) {
        if (!latestEscapingUser || latestEscapingUser->isBeforeInBlock(escapingUser)) {
          latestEscapingUser = escapingUser;
        }
        users.insert(escapingUser);
      }
    }
    latestEscapingUser->emitRemark("Reordering operations...");
    reorderOperations(earliestInput, inputs, users);
    earliestInput->getBlock()->recomputeOpOrder();
    latestEscapingUser->emitRemark("Reordering done.");
  }
  for (auto& entry : escapingUsers) {
    std::sort(std::begin(entry.second), std::end(entry.second), [&](Operation* lhs, Operation* rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
  }

  // Compute insertion points.
  DenseMap<ValueVector*, Value> currentInsertionPoint;
  DenseMap<Value, ValueVector*> lastVectorAfterValue;
  Value currentLatest;
  for (auto* vector : order) {
    if (vector->isLeaf()) {
      auto const& element = latestElement(vector);
      currentInsertionPoint[vector] = element;
      auto pair = lastVectorAfterValue.try_emplace(element, vector);
      if (!pair.second) {
        insertionPoints[vector] = pair.first->second;
        pair.first->getSecond() = vector;
      }
    } else {
      Value latestOperand;
      for (size_t i = 0; i < vector->numOperands(); ++i) {
        auto* operand = vector->getOperand(i);
        if (willBeFolded(operand)) {
          continue;
        }
        auto const& nextLatest = currentInsertionPoint[operand];
        if (!latestOperand || !later(latestOperand, nextLatest)) {
          latestOperand = nextLatest;
        }
      }
      if (latestOperand) {
        insertionPoints[vector] = lastVectorAfterValue[latestOperand];
        currentInsertionPoint[vector] = latestOperand;
        lastVectorAfterValue[latestOperand] = vector;
      } else {
        insertionPoints[vector] = lastVectorAfterValue[currentLatest];
        currentInsertionPoint[vector] = currentLatest;
        lastVectorAfterValue[currentLatest] = vector;
      }
    }
    if (!currentLatest || later(currentInsertionPoint[vector], currentLatest)) {
      currentLatest = currentInsertionPoint[vector];
    }
  }
}

void ConversionManager::setInsertionPointFor(ValueVector* vector) const {
  if (!insertionPoints.count(vector)) {
    rewriter.setInsertionPointAfterValue(latestElement(vector));
  } else {
    rewriter.setInsertionPointAfterValue(creationData.lookup(insertionPoints.lookup(vector)).operation);
  }
}

void ConversionManager::update(ValueVector* vector, Value const& operation, ElementFlag flag) {
  assert(!wasConverted(vector) && "vector has been converted already");
  creationData[vector].operation = operation;
  creationData[vector].flag = flag;
}

bool ConversionManager::wasConverted(ValueVector* vector) const {
  return creationData.count(vector);
}

Value ConversionManager::getValue(ValueVector* vector) const {
  assert(wasConverted(vector) && "vector has not yet been converted");
  return creationData.lookup(vector).operation;
}

ElementFlag ConversionManager::getElementFlag(ValueVector* vector) const {
  assert(wasConverted(vector) && "vector has not yet been converted");
  return creationData.lookup(vector).flag;
}

ArrayRef<ValueVector*> ConversionManager::conversionOrder() const {
  return order;
}

bool ConversionManager::hasEscapingUsers(Value const& value) const {
  return escapingUsers.count(value) && !escapingUsers.lookup(value).empty();
}

Operation* ConversionManager::getEarliestEscapingUser(Value const& value) const {
  assert(hasEscapingUsers(value) && "value does not have any escaping users");
  return escapingUsers.lookup(value).front();
}

void ConversionManager::replaceEscapingUsersWith(Value const& value, Value const& newValue) {
  assert(hasEscapingUsers(value) && "value does not have any escaping users");
  for (auto* escapingUser : escapingUsers.lookup(value)) {
    size_t index = 0;
    for (auto const& operand : escapingUser->getOperands()) {
      if (operand == value) {
        break;
      }
      ++index;
    }
    escapingUser->setOperand(index, newValue);
  }
}

Value ConversionManager::getConstant(Location const& loc, Attribute const& attribute) {
  return folder.getOrCreateConstant(rewriter, &attribute.getDialect(), attribute, attribute.getType(), loc);
}
