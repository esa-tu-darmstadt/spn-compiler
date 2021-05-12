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

  Value latest(Value const& lhs, Value const& rhs) {
    if (lhs.isa<BlockArgument>()) {
      return rhs.isa<BlockArgument>() ? lhs : rhs;
    } else if (rhs.isa<BlockArgument>()) {
      return lhs;
    }
    return lhs.getDefiningOp()->isBeforeInBlock(rhs.getDefiningOp()) ? rhs : lhs;
  }

  Value latestElement(NodeVector* vector) {
    Value latestElement = nullptr;
    for (auto const& value : *vector) {
      if (!latestElement) {
        latestElement = value;
        continue;
      }
      latestElement = latest(latestElement, value);
    }
    return latestElement;
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
    // Sort operations in between the earliest escaping user & the latest input.
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

  void computeOrder(SLPNode* root, SmallVectorImpl<NodeVector*>& order) {
    for (auto* node : graph::postOrder(root)) {
      for (size_t i = node->numVectors(); i-- > 0;) {
        order.emplace_back(node->getVector(i));
      }
    }
  }

}

ConversionManager::ConversionManager(SLPNode* root) : folder{root->getValue(0, 0).getContext()} {
  computeOrder(root, order);
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
        if (vector->numOperands() == 0) {
          if (!earliestInput || elementOp->isBeforeInBlock(earliestInput)) {
            earliestInput = elementOp;
          }
          inputs.insert(elementOp);
        }
      }
    }
  }
  if (!earliestInput) {
    // No non-block-argument-input => SLP graph insertion point can be at the beginning of the basic block (i.e. null).
    return;
  }

  Operation* earliestEscapingUser = nullptr;
  Operation* latestEscapingUser = nullptr;
  SmallPtrSet<Operation*, 32> users;
  for (auto& entry : escapingUsers) {
    for (auto* escapingUser : entry.second) {
      if (!earliestEscapingUser || escapingUser->isBeforeInBlock(earliestEscapingUser)) {
        earliestEscapingUser = escapingUser;
      }
      if (!latestEscapingUser || latestEscapingUser->isBeforeInBlock(escapingUser)) {
        latestEscapingUser = escapingUser;
      }
      users.insert(escapingUser);
    }
  }
  earliestEscapingUser->emitRemark("Reordering operations...");
  reorderOperations(earliestInput, inputs, users);
  earliestEscapingUser->emitRemark("Reordering done.");
  insertionPoint = std::make_pair(earliestEscapingUser, true);
}

void ConversionManager::setInsertionPointFor(NodeVector* vector, PatternRewriter& rewriter) const {
  Operation* latestOperandOp = nullptr;
  for (size_t i = 0; i < vector->numOperands(); ++i) {
    auto* operand = vector->getOperand(i);
    assert(vectorData.lookup(operand).operation.hasValue() && "operand has not yet been converted");
    auto* operandOp = vectorData.lookup(operand).operation->getDefiningOp();
    if (!latestOperandOp || latestOperandOp->isBeforeInBlock(operandOp)) {
      latestOperandOp = operandOp;
    }
  }
  if (latestOperandOp) {
    rewriter.setInsertionPointAfter(latestOperandOp);
  } else if (!insertionPoint.hasValue()) {
    rewriter.setInsertionPointToStart(vector->getElement(0).getParentBlock());
  } else {
    auto const& latest = latestElement(vector);
    if (auto* latestElementOp = latest.getDefiningOp()) {
      rewriter.setInsertionPointAfter(latestElementOp);
      return;
    }
    if (insertionPoint->second) {
      rewriter.setInsertionPoint(insertionPoint->first);
    } else {
      rewriter.setInsertionPointAfter(insertionPoint->first);
    }
  }
}

void ConversionManager::update(NodeVector* vector, Value const& operation, ElementFlag const& flag) {
  assert(!vectorData[vector].operation.hasValue() && !vectorData[vector].flag.hasValue()
             && "vector has been converted already");
  vectorData[vector].operation = operation;
  vectorData[vector].flag = flag;
  if (insertionPoint->first->isBeforeInBlock(operation.getDefiningOp())) {
    insertionPoint = std::make_pair(operation.getDefiningOp(), false);
  }
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

ArrayRef<NodeVector*> ConversionManager::conversionOrder() const {
  return order;
}

bool ConversionManager::hasEscapingUsers(Value const& value) const {
  return escapingUsers.count(value) && !escapingUsers.lookup(value).empty();
}

Operation* ConversionManager::getEarliestEscapingUser(Value const& value) const {
  assert(hasEscapingUsers(value) && "value does not have any escaping users");
  Operation* earliestEscapingUser = nullptr;
  for (auto* escapingUser : escapingUsers.lookup(value)) {
    if (!earliestEscapingUser || escapingUser->isBeforeInBlock(earliestEscapingUser)) {
      earliestEscapingUser = escapingUser;
    }
  }
  return earliestEscapingUser;
}

Value ConversionManager::getConstant(Location const& loc, Attribute const& attribute, PatternRewriter& rewriter) {
  return folder.getOrCreateConstant(rewriter, &attribute.getDialect(), attribute, attribute.getType(), loc);
}
