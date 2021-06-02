//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/GraphConversion.h"
#include "LoSPNtoCPU/Vectorization/SLP/CostModel.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

namespace llvm {

  template<>
  struct GraphTraits<DependencyGraph> {
    typedef Superword* NodeRef;
    typedef SmallPtrSetIterator<NodeRef> ChildIteratorType;
    static NodeRef getEntryNode(DependencyGraph const&) {
      llvm_unreachable("TODO");
    }
    static ChildIteratorType child_begin(NodeRef) {
      llvm_unreachable("TODO");
    }
    static ChildIteratorType child_end(NodeRef) {
      llvm_unreachable("TODO");
    }

    typedef SmallPtrSetIterator<NodeRef> nodes_iterator;
    static nodes_iterator nodes_begin(DependencyGraph* G) {
      llvm_unreachable("TODO");
    }
    static nodes_iterator nodes_end(DependencyGraph* G) {
      llvm_unreachable("TODO");
    }

    typedef std::pair<NodeRef, NodeRef> EdgeRef;
    typedef SmallPtrSetIterator<EdgeRef> ChildEdgeIteratorType;
    static ChildEdgeIteratorType child_edge_begin(NodeRef) {
      llvm_unreachable("TODO");
    }
    static ChildEdgeIteratorType child_edge_end(NodeRef) {
      llvm_unreachable("TODO");
    }
    static NodeRef edge_dest(EdgeRef) {
      llvm_unreachable("TODO");
    }

    static unsigned size(DependencyGraph* G) {
      llvm_unreachable("TODO");
    }
  };

}

// === ConversionState === //

bool ConversionState::alreadyComputed(Superword* superword) const {
  return computedSuperwords.contains(superword);
}

bool ConversionState::alreadyComputed(Value const& value) const {
  return computedScalarValues.contains(value);
}

bool ConversionState::isExtractionProfitable(Value const& value) const {
  return profitableExtractions.contains(value);
}

void ConversionState::markComputed(Superword* superword) {
  computedSuperwords.insert(superword);
  for (auto* operand : superword->getOperands()) {
    assert (alreadyComputed(operand) && "computing vector before its operands");
  }
}

void ConversionState::markComputed(Value const& value) {
  computedScalarValues.insert(value);
  if (auto* definingOp = value.getDefiningOp()) {
    for (auto const& operand : definingOp->getOperands()) {
      markComputed(operand);
    }
  }
}

void ConversionState::markExtractionProfitable(Value const& value) {
  profitableExtractions.insert(value);
}

ValuePosition ConversionState::getWordContainingValue(Value const& value) const {
  return extractableScalarValues.lookup(value);
}

// === ConversionPlan === //

ConversionPlan::ConversionPlan(std::shared_ptr<ConversionState> conversionState) : conversionState{
    std::move(conversionState)} {}

void ConversionPlan::addConversionStep(Superword* superword, SLPVectorizationPattern* pattern) {
  for (auto const& element : scalarVisitor.getRequiredScalarValues(pattern, superword)) {
    if (conversionState->alreadyComputed(element)) {
      continue;
    }
    if (costModel->isExtractionProfitable(element)) {
      llvm_unreachable("TODO: create extraction step");
    }
    conversionState->markComputed(element);
  }
  plan.emplace_back(superword, pattern);
}

// Helper functions in anonymous namespace.
namespace {

  SmallVector<Superword*> computeOrder(Superword* root) {
    DenseMap<Superword*, unsigned> depths;
    depths[root] = 0;
    SmallVector<Superword*> worklist{root};
    while (!worklist.empty()) {
      auto* superword = worklist.pop_back_val();
      for (auto* operand : superword->getOperands()) {
        auto operandDepth = depths[superword] + 1;
        if (depths[operand] < operandDepth) {
          depths[operand] = operandDepth;
          worklist.emplace_back(operand);
        }
      }
    }
    SmallVector<Superword*> order;
    for (auto const& entry: depths) {
      order.emplace_back(entry.first);
    }
    llvm::sort(std::begin(order), std::end(order), [&](Superword* lhs, Superword* rhs) {
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

  Value latestElement(Superword* superword) {
    Value latestElement;
    for (auto const& value : *superword) {
      if (!latestElement || later(value, latestElement)) {
        latestElement = value;
      }
    }
    return latestElement;
  }

  bool willBeFolded(Superword* superword) {
    return std::all_of(std::begin(*superword), std::end(*superword), [&](Value const& element) {
      return element.getDefiningOp() && element.getDefiningOp()->hasTrait<OpTrait::ConstantLike>();
    });
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

void ConversionManager::initConversion(Superword* root) {
  order.assign(computeOrder(root));

  SmallPtrSet<Operation*, 32> inputs;
  Operation* earliestInput = nullptr;

  // Remove all temporary data from previous graphs.
  escapingUsers.clear();
  insertionPoints.clear();

  for (auto* superword : order) {
    for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
      auto const& element = superword->getElement(lane);
      if (auto* elementOp = element.getDefiningOp()) {
        if (!escapingUsers.count(element)) {
          escapingUsers[element].assign(std::begin(element.getUsers()), std::end(element.getUsers()));
        }
        if (superword->isLeaf()) {
          if (!earliestInput || elementOp->isBeforeInBlock(earliestInput)) {
            earliestInput = elementOp;
          }
          inputs.insert(elementOp);
        } else {
          superwordPositions.try_emplace(element, superword, lane);
          for (size_t i = 0; i < superword->numOperands(); ++i) {
            auto const& operand = superword->getOperand(i)->getElement(lane);
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
  DenseMap<Superword*, Value> currentInsertionPoint;
  DenseMap<Value, Superword*> lastWordAfterValue;
  Value currentLatest;
  for (auto* superword : order) {
    if (superword->isLeaf()) {
      auto const& element = latestElement(superword);
      currentInsertionPoint[superword] = element;
      auto pair = lastWordAfterValue.try_emplace(element, superword);
      if (!pair.second) {
        insertionPoints[superword] = pair.first->second;
        pair.first->getSecond() = superword;
      }
    } else {
      Value latestOperand;
      for (size_t i = 0; i < superword->numOperands(); ++i) {
        auto* operand = superword->getOperand(i);
        if (willBeFolded(operand)) {
          continue;
        }
        auto const& nextLatest = currentInsertionPoint[operand];
        if (!latestOperand || !later(latestOperand, nextLatest)) {
          latestOperand = nextLatest;
        }
      }
      if (latestOperand) {
        insertionPoints[superword] = lastWordAfterValue[latestOperand];
        currentInsertionPoint[superword] = latestOperand;
        lastWordAfterValue[latestOperand] = superword;
      } else {
        insertionPoints[superword] = lastWordAfterValue[currentLatest];
        currentInsertionPoint[superword] = currentLatest;
        lastWordAfterValue[currentLatest] = superword;
      }
    }
    if (!currentLatest || later(currentInsertionPoint[superword], currentLatest)) {
      currentLatest = currentInsertionPoint[superword];
    }
  }
}

void ConversionManager::setInsertionPointFor(Superword* superword) const {
  if (!insertionPoints.count(superword)) {
    rewriter.setInsertionPointAfterValue(latestElement(superword));
  } else {
    rewriter.setInsertionPointAfterValue(creationData.lookup(insertionPoints.lookup(superword)).operation);
  }
}

void ConversionManager::update(Superword* superword, Value const& operation, ElementFlag flag) {
  assert(!wasConverted(superword) && "superword has been converted already");
  creationData[superword].operation = operation;
  creationData[superword].flag = flag;
}

bool ConversionManager::wasConverted(Superword* superword) const {
  return creationData.count(superword);
}

Value ConversionManager::getValue(Superword* superword) const {
  assert(wasConverted(superword) && "superword has not yet been converted");
  return creationData.lookup(superword).operation;
}

ElementFlag ConversionManager::getElementFlag(Superword* superword) const {
  superword->begin()->getDefiningOp()->getParentOfType<FuncOp>().dump();
  dumpSuperword(*superword);
  assert(wasConverted(superword) && "superword has not yet been converted");
  return creationData.lookup(superword).flag;
}

ArrayRef<Superword*> ConversionManager::conversionOrder() const {
  return order;
}

bool ConversionManager::hasEscapingUsers(Value const& value) const {
  return escapingUsers.count(value) && !escapingUsers.lookup(value).empty();
}

Value ConversionManager::getOrCreateConstant(Location const& loc, Attribute const& attribute) {
  return folder.getOrCreateConstant(rewriter, &attribute.getDialect(), attribute, attribute.getType(), loc);
}

Value ConversionManager::getOrExtractValue(Value const& value) {
  auto const& wordPosition = superwordPositions.lookup(value);
  if (!wordPosition.first) {
    return value;
  }
  auto elementFlag = getElementFlag(wordPosition.first);
  if (elementFlag == ElementFlag::KeepAll || (elementFlag == ElementFlag::KeepFirst && wordPosition.second == 0)) {
    return value;
  }
  auto const& source = getValue(wordPosition.first);
  auto const& pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) wordPosition.second));
  return rewriter.create<vector::ExtractElementOp>(value.getLoc(), source, pos);
}

void ConversionManager::createExtractionFor(Value const& value) {
  assert(hasEscapingUsers(value) && "value does not have escaping uses");
  auto const& wordPosition = superwordPositions.lookup(value);
  auto const& source = getValue(wordPosition.first);
  auto pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) wordPosition.second));
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
