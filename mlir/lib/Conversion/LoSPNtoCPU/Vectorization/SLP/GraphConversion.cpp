//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/GraphConversion.h"
#include "LoSPNtoCPU/Vectorization/SLP/CostModel.h"
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

void ConversionState::markComputed(Superword* superword) {
  for (auto* operand : superword->getOperands()) {
    assert (alreadyComputed(operand) && "computing vector before its operands");
  }
  computedSuperwords.insert(superword);
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    extractableScalarValues.try_emplace(superword->getElement(lane), superword, lane);
  }
  for (auto const& callback : vectorCallbacks) {
    callback(superword);
  }
}

void ConversionState::markComputed(Value const& value) {
  computedScalarValues.insert(value);
  if (auto* definingOp = value.getDefiningOp()) {
    for (auto const& operand : definingOp->getOperands()) {
      markComputed(operand);
    }
  }
  for (auto const& callback : scalarCallbacks) {
    callback(value);
  }
}

ValuePosition ConversionState::getWordContainingValue(Value const& value) const {
  return extractableScalarValues.lookup(value);
}

void ConversionState::addVectorCallback(std::function<void(Superword*)> callback) {
  vectorCallbacks.emplace_back(std::move(callback));
}

void ConversionState::addScalarCallback(std::function<void(Value)> callback) {
  scalarCallbacks.emplace_back(std::move(callback));
}

void ConversionState::addExtractionCallback(std::function<void(Value)> callback) {
  extractionCallbacks.emplace_back(std::move(callback));
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

  Operation* earliestNonConstOperation(Block* block) {
    for (auto& op : *block) {
      if (!op.hasTrait<OpTrait::ConstantLike>()) {
        return &op;
      }
    }
    llvm_unreachable("a block consisting of constant operations only should not need to be vectorized");
  }

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
      // This comparison maximizes the re-use potential of non-leaf elements in leaf nodes through extractions.
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

  void reorderOperations(Operation* earliestInput, Block* block, SmallPtrSetImpl<Operation*> const& escapingUsers) {
    DenseMap<Operation*, unsigned> depths;
    SmallVector<Operation*> worklist;
    for (auto* user : escapingUsers) {
      worklist.emplace_back(user);
      depths[user] = 0;
    }
    unsigned maxDepth = 0;
    while (!worklist.empty()) {
      auto* currentOp = worklist.pop_back_val();
      for (auto const& operand : currentOp->getOperands()) {
        if (auto* operandOp = operand.getDefiningOp()) {
          if (earliestInput && operandOp->isBeforeInBlock(earliestInput)) {
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

    Operation* latestOp = earliestInput;
    for (unsigned depth = 0; depth <= maxDepth; ++depth) {
      auto const& ops = opsSortedByDepth[depth];
      for (auto* op : ops) {
        // Earliest input == block argument?
        if (!latestOp) {
          latestOp = earliestNonConstOperation(block)->getPrevNode();
          if (!latestOp) {
            op->moveBefore(block, block->begin());
            latestOp = op;
            continue;
          }
        }
        op->moveAfter(latestOp);
        latestOp = op;
      }
    }
  }

}

ConversionManager::ConversionManager(PatternRewriter& rewriter, std::shared_ptr<ConversionState> conversionState)
    : conversionState{std::move(conversionState)}, rewriter{rewriter}, folder{rewriter.getContext()} {}

void ConversionManager::initConversion(Superword* root, Block* block) {
  order.assign(computeOrder(root));

  // Remove all temporary data from previous graphs.
  escapingUsers.clear();
  insertionPoints.clear();

  // Gather escaping users.
  Operation* earliestInput = nullptr;
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
        } else {
          for (size_t i = 0; i < superword->numOperands(); ++i) {
            auto const& operand = superword->getOperand(i)->getElement(lane);
            auto& users = escapingUsers[operand];
            users.erase(std::remove(std::begin(users), std::end(users), elementOp), std::end(users));
          }
        }
      }
    }
  }
  SmallPtrSet<Operation*, 32> users;
  for (auto& entry : escapingUsers) {
    users.insert(std::begin(entry.second), std::end(entry.second));
  }

  assert(!users.empty() && "trying to vectorize dead function");
  reorderOperations(earliestInput, block, users);

  // Sort escaping users so that we can create the extraction operation right in front of the first one.
  for (auto& entry : escapingUsers) {
    llvm::sort(std::begin(entry.second), std::end(entry.second), [&](Operation* lhs, Operation* rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
  }

  // Compute insertion points.
  for (auto* superword : order) {
    if (superword->isLeaf()) {
      auto const& latest = latestElement(superword);
      if (auto* latestOp = latest.getDefiningOp()) {
        insertionPoints[superword] = latestOp->getNextNode();
      } else {
        insertionPoints[superword] = earliestNonConstOperation(block);
      }
    } else {
      Operation* latestOperand = nullptr;
      for (size_t i = 0; i < superword->numOperands(); ++i) {
        auto* operand = superword->getOperand(i);
        if (willBeFolded(operand)) {
          continue;
        }
        auto* nextLatest = insertionPoints[operand];
        if (!latestOperand || latestOperand->isBeforeInBlock(nextLatest)) {
          latestOperand = nextLatest;
        }
      }
      // Make sure that if vectorization patterns fail, broadcast & insert patterns can still be applied.
      // Here, the latest element always has a defining op (otherwise the superword would be a leaf).
      auto latestOp = latestElement(superword).getDefiningOp();
      if (latestOp->isBeforeInBlock(latestOperand)) {
        insertionPoints[superword] = latestOperand;
      } else {
        insertionPoints[superword] = latestOp->getNextNode();
      }
    }
  }
}

void ConversionManager::setInsertionPointFor(Superword* superword) const {
  auto* insertionPoint = insertionPoints.lookup(superword);
  if (insertionPoint) {
    rewriter.setInsertionPoint(insertionPoint);
  } else {
    rewriter.setInsertionPointToEnd(superword->getElement(0).getParentBlock());
  }
}

bool ConversionManager::wasConverted(Superword* superword) const {
  return creationData.count(superword);
}

void ConversionManager::update(Superword* superword, Value const& operation, ElementFlag flag) {
  assert(!wasConverted(superword) && "superword has been converted already");
  creationData[superword].operation = operation;
  creationData[superword].flag = flag;
}

Value ConversionManager::getValue(Superword* superword) const {
  assert(wasConverted(superword) && "superword has not yet been converted");
  return creationData.lookup(superword).operation;
}

ElementFlag ConversionManager::getElementFlag(Superword* superword) const {
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
  auto const& wordPosition = conversionState->getWordContainingValue(value);
  if (!wordPosition.superword) {
    return value;
  }
  auto elementFlag = getElementFlag(wordPosition.superword);
  if (elementFlag == ElementFlag::KeepAll || (elementFlag == ElementFlag::KeepFirst && wordPosition.index == 0)) {
    return value;
  }
  auto const& source = getValue(wordPosition.superword);
  auto const& pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) wordPosition.index));
  return rewriter.create<vector::ExtractElementOp>(value.getLoc(), source, pos);
}

void ConversionManager::createExtractionFor(Value const& value) {
  assert(hasEscapingUsers(value) && "value does not have escaping uses");
  auto const& wordPosition = conversionState->getWordContainingValue(value);
  auto const& source = getValue(wordPosition.superword);
  auto pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) wordPosition.index));
  rewriter.setInsertionPoint(escapingUsers.lookup(value).front());
  Value extractOp = rewriter.create<vector::ExtractElementOp>(value.getLoc(), source, pos);
  for (auto* escapingUser : escapingUsers.lookup(value)) {
    size_t index = 0;
    for (auto const& operand : escapingUser->getOperands()) {
      if (operand == value) {
        if (operand.getType().isa<LogType>()) {
          extractOp = rewriter.create<SPNAttachLog>(value.getLoc(), extractOp, extractOp.getType());
        }
        break;
      }
      ++index;
    }
    escapingUser->setOperand(index, extractOp);
  }
  escapingUsers.erase(value);
}