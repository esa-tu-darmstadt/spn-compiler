//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/GraphConversion.h"
#include "mlir/Dialect/Vector/VectorOps.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// Helper functions in anonymous namespace.
namespace {

  SmallVector<ValueVector*> computeOrder(ValueVector* root) {
    DenseMap<ValueVector*, unsigned> depths;
    depths[root] = 0;
    SmallVector<ValueVector*> worklist{root};
    while (!worklist.empty()) {
      auto* vector = worklist.pop_back_val();
      for (auto* operand : vector->getOperands()) {
        auto operandDepth = depths[vector] + 1;
        if (depths[operand] < operandDepth) {
          depths[operand] = operandDepth;
          worklist.emplace_back(operand);
        }
      }
    }
    SmallVector<ValueVector*> order;
    for (auto const& entry: depths) {
      order.emplace_back(entry.first);
    }
    llvm::sort(std::begin(order), std::end(order), [&](ValueVector* lhs, ValueVector* rhs) {
      // This additional comparison maximizes the re-use potential of leaf vectors.
      if (depths[lhs] == depths[rhs]) {
        return !lhs->isLeaf() && rhs->isLeaf();
      }
      return depths[lhs] > depths[rhs];
    });
    return order;
  }

  bool later(Value const& lhs, Value const& rhs) {
    if (lhs == rhs || lhs.isa<BlockArgument>()) {
      return false;
    } else if (rhs.isa<BlockArgument>()) {
      return true;
    }
    return rhs.getDefiningOp()->isBeforeInBlock(lhs.getDefiningOp());
  }

  Value latestElement(ValueVector* vector) {
    Value latestElement;
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
      llvm::sort(std::begin(ops), std::end(ops), [&](Operation* lhs, Operation* rhs) {
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

ConversionManager::ConversionManager(PatternRewriter& rewriter) : rewriter{rewriter}, folder{rewriter.getContext()} {}

void ConversionManager::initConversion(ValueVector* root) {
  order.assign(computeOrder(root));

  SmallPtrSet<Operation*, 32> inputs;
  Operation* earliestInput = nullptr;

  // Remove all temporary data from previous graphs.
  escapingUsers.clear();
  insertionPoints.clear();

  for (auto* vector : order) {
    for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
      auto const& element = vector->getElement(lane);
      if (auto* elementOp = element.getDefiningOp()) {
        if (!escapingUsers.count(element)) {
          escapingUsers[element].assign(std::begin(element.getUsers()), std::end(element.getUsers()));
        }
        if (vector->isLeaf()) {
          if (!earliestInput || elementOp->isBeforeInBlock(earliestInput)) {
            earliestInput = elementOp;
          }
          inputs.insert(elementOp);
        } else {
          vectorPositions.try_emplace(element, vector, lane);
          for (size_t i = 0; i < vector->numOperands(); ++i) {
            auto const& operand = vector->getOperand(i)->getElement(lane);
            auto& users = escapingUsers[operand];
            users.erase(std::remove(std::begin(users), std::end(users), elementOp), std::end(users));
          }
        }
      }
    }
  }
  if (earliestInput) {
    Operation* latestEscapingUser = nullptr;
    SmallPtrSet<Operation*, 32> users;
    for (auto& entry : escapingUsers) {
      for (auto* escapingUser : entry.second) {
        if (users.contains(escapingUser)) {
          continue;
        }
        if (!latestEscapingUser || latestEscapingUser->isBeforeInBlock(escapingUser)) {
          latestEscapingUser = escapingUser;
        }
        users.insert(escapingUser);
      }
    }
    reorderOperations(earliestInput, inputs, users);
  }

  // Sort escaping users so that we can create the extraction operation right in front of the first one.
  for (auto& entry : escapingUsers) {
    auto& users = entry.second;
    llvm::sort(std::begin(users), std::end(users), [&](Operation* lhs, Operation* rhs) {
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

Value ConversionManager::getOrCreateConstant(Location const& loc, Attribute const& attribute) {
  return folder.getOrCreateConstant(rewriter, &attribute.getDialect(), attribute, attribute.getType(), loc);
}

Value ConversionManager::getOrExtractValue(Value const& value) {
  auto const& vectorPosition = vectorPositions.lookup(value);
  if (!vectorPosition.first) {
    return value;
  }
  auto elementFlag = getElementFlag(vectorPosition.first);
  if (elementFlag == ElementFlag::KeepAll || (elementFlag == ElementFlag::KeepFirst && vectorPosition.second == 0)) {
    return value;
  }
  auto const& source = getValue(vectorPosition.first);
  auto const& pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) vectorPosition.second));
  return rewriter.create<vector::ExtractElementOp>(value.getLoc(), source, pos);
}

void ConversionManager::createExtractionFor(Value const& value) {
  assert(hasEscapingUsers(value) && "value does not have escaping uses");
  auto const& vectorPosition = vectorPositions.lookup(value);
  auto const& source = getValue(vectorPosition.first);
  auto pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) vectorPosition.second));
  rewriter.setInsertionPoint(escapingUsers.lookup(value).front());
  auto extractOp = rewriter.create<vector::ExtractElementOp>(value.getLoc(), source, pos);
  for (auto* escapingUser : escapingUsers.lookup(value)) {
    size_t index = 0;
    for (auto const& operand : escapingUser->getOperands()) {
      if (operand == value) {
        break;
      }
      ++index;
    }
    escapingUser->setOperand(index, extractOp.result());
  }
  escapingUsers.erase(value);
}
