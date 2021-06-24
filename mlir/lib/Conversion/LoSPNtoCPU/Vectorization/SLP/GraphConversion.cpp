//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/GraphConversion.h"
#include "LoSPNtoCPU/Vectorization/SLP/CostModel.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "mlir/Dialect/Vector/VectorOps.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

// === ConversionState === //

bool ConversionState::alreadyComputed(Superword* superword) const {
  return computedSuperwords.count(superword);
}

bool ConversionState::alreadyComputed(Value const& value) const {
  return computedScalarValues.contains(value) || extractedScalarValues.contains(value);
}

bool ConversionState::isDead(Operation* op) const {
  return deadOps.contains(op);
}

void ConversionState::markComputed(Value const& value) {
  if (computedScalarValues.insert(value).second) {
    if (auto* definingOp = value.getDefiningOp()) {
      for (auto const& operand : definingOp->getOperands()) {
        markComputed(operand);
      }
    }
    for (auto const& callback : scalarCallbacks) {
      callback(value);
    }
  }
}

void ConversionState::markComputed(Superword* superword, Value value) {
  assert(!alreadyComputed(superword) && "superword has already been converted");
  for (auto* operand : superword->getOperands()) {
    assert(alreadyComputed(operand) && "computing vector before its operands");
  }
  computedSuperwords[superword] = value;
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    extractableScalarValues.try_emplace(superword->getElement(lane), superword, lane);
  }
  for (auto const& callback : vectorCallbacks) {
    callback(superword);
  }
}

void ConversionState::markExtracted(Value const& value) {
  if (extractedScalarValues.insert(value).second) {
    for (auto const& callback : extractionCallbacks) {
      callback(value);
    }
  }
}

void ConversionState::markDead(Operation* op) {
  deadOps.insert(op);
}

Value ConversionState::getValue(Superword* superword) const {
  assert(alreadyComputed(superword) && "superword has not yet been converted");
  return computedSuperwords.lookup(superword);
}

ValuePosition ConversionState::getWordContainingValue(Value const& value) const {
  return extractableScalarValues.lookup(value);
}

SmallPtrSet<Operation*, 32> const& ConversionState::getDeadOps() const {
  return deadOps;
}

void ConversionState::undoChanges() const {
  for (auto const& callback : scalarUndoCallbacks) {
    for (auto value : computedScalarValues) {
      callback(value);
    }
  }
  for (auto const& callback : vectorUndoCallbacks) {
    for (auto const& entry : computedSuperwords) {
      callback(entry.first);
    }
  }
  for (auto const& callback : extractionUndoCallbacks) {
    for (auto value : extractedScalarValues) {
      callback(value);
    }
  }
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

void ConversionState::addVectorUndoCallback(std::function<void(Superword*)> callback) {
  vectorUndoCallbacks.emplace_back(std::move(callback));
}

void ConversionState::addScalarUndoCallback(std::function<void(Value)> callback) {
  scalarUndoCallbacks.emplace_back(std::move(callback));
}

void ConversionState::addExtractionUndoCallback(std::function<void(Value)> callback) {
  extractionUndoCallbacks.emplace_back(std::move(callback));
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
      // This comparison maximizes the re-use potential of non-leaf elements in leaf nodes through extractions.
      if (depths[lhs] == depths[rhs]) {
        return !lhs->isLeaf() && rhs->isLeaf();
      }
      return depths[lhs] > depths[rhs];
    });
    return order;
  }

}

ConversionManager::ConversionManager(PatternRewriter& rewriter, std::shared_ptr<CostModel> costModel) : costModel{
    std::move(costModel)}, rewriter{rewriter}, folder{rewriter.getContext()} {}

void ConversionManager::initConversion(Superword* root, Block* block, std::shared_ptr<ConversionState> graphState) {
  // Clear all temporary data.
  escapingUsers.clear();
  originalOperations.clear();
  originalOperands.clear();
  conversionState = std::move(graphState);
  order.assign(computeOrder(root));
  graphBlock = block;
  // Store original block state for undoing graph conversions.
  block->walk([&](Operation* op) {
    originalOperations.emplace_back(op);
  });
  // Gather escaping users.
  Operation* earliestInput = nullptr;
  for (auto* superword : order) {
    for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
      auto const& element = superword->getElement(lane);
      if (auto* elementOp = element.getDefiningOp()) {
        if (elementOp->hasTrait<OpTrait::ConstantLike>()) {
          continue;
        }
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
  // Sort escaping users so that we can create the extraction operation right in front of the first one.
  for (auto& entry : escapingUsers) {
    llvm::sort(std::begin(entry.second), std::end(entry.second), [&](Operation* lhs, Operation* rhs) {
      lhs->dump();
      rhs->dump();
      return lhs->isBeforeInBlock(rhs);
    });
  }
}

void ConversionManager::finishConversion() {
  // Sort operations topologically.
  DenseMap<Operation*, unsigned> depths;
  llvm::SmallSetVector<Operation*, 32> worklist;
  graphBlock->walk<WalkOrder::PreOrder>([&](Operation* op) {
    depths[op] = 0;
    worklist.insert(op);
  });
  unsigned maxDepth = 0;
  while (!worklist.empty()) {
    auto* currentOp = worklist.pop_back_val();
    for (auto const& operand : currentOp->getOperands()) {
      if (auto* operandOp = operand.getDefiningOp()) {
        unsigned operandDepth = depths[currentOp] + 1;
        if (operandDepth > depths[operandOp]) {
          depths[operandOp] = operandDepth;
          maxDepth = std::max(maxDepth, operandDepth);
          worklist.insert(operandOp);
        }
      }
    }
  }
  SmallVector<SmallVector<Operation*>> opsSortedByDepth{maxDepth + 1};
  for (auto const& entry : depths) {
    opsSortedByDepth[maxDepth - entry.second].emplace_back(entry.first);
  }
  for (auto& ops: opsSortedByDepth) {
    llvm::sort(std::begin(ops), std::end(ops), [&](Operation* lhs, Operation* rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
  }
  Operation* lastOp = nullptr;
  for (auto const& ops : opsSortedByDepth) {
    for (auto* op : ops) {
      if (lastOp) {
        op->moveAfter(lastOp);
      } else {
        op->moveBefore(graphBlock, graphBlock->begin());
      }
      lastOp = op;
    }
  }
}

void ConversionManager::undoConversion() {
  Operation* lastOp = nullptr;
  for (auto* op : originalOperations) {
    auto operands = originalOperands.lookup(op);
    if (!operands.empty()) {
      op->setOperands(operands);
    }
    if (lastOp) {
      op->moveAfter(lastOp);
    } else {
      op->moveBefore(graphBlock, graphBlock->begin());
    }
    lastOp = op;
  }
  // Move all created operations to a trash block, so that they can be destroyed *immediately* instead of at the end
  // of the conversion process. We don't want to carry thousands or millions of erased operations with us to the next
  // vectorization attempt.
  Block* trashBlock = rewriter.createBlock(graphBlock);
  graphBlock->moveBefore(trashBlock);
  while (auto* nextNode = lastOp->getNextNode()) {
    nextNode->moveBefore(trashBlock, trashBlock->end());
    nextNode->dropAllReferences();
  }
  rewriter.eraseBlock(trashBlock);
  folder.clear();
  conversionState->undoChanges();
}

ArrayRef<Superword*> ConversionManager::conversionOrder() const {
  return order;
}

void ConversionManager::setupConversionFor(Superword* superword, SLPVectorizationPattern const* pattern) {
  rewriter.setInsertionPointToEnd(graphBlock);
  // Create extractions if needed.
  auto scalarInputs = leafVisitor.getRequiredScalarValues(pattern, superword);
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    auto const& element = superword->getElement(lane);
    if (std::find(std::begin(scalarInputs), std::end(scalarInputs), element) != std::end(scalarInputs)) {
      superword->setElement(lane, getOrExtractValue(element));
    }
  }
}

void ConversionManager::update(Superword* superword,
                               Value const& operation,
                               SLPVectorizationPattern const* appliedPattern) {
  conversionState->markComputed(superword, operation);
  auto scalarInputs = leafVisitor.getRequiredScalarValues(appliedPattern, superword);
  for (auto const& scalarInput : scalarInputs) {
    conversionState->markComputed(scalarInput);
  }
  // Create vector extractions for escaping uses.
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    auto const& element = superword->getElement(lane);
    if (conversionState->alreadyComputed(element)) {
      continue;
    }
    if (hasEscapingUsers(element)) {
      Value extractOp = getOrExtractValue(element);
      if (extractOp.getType().isa<LogType>()) {
        extractOp = rewriter.create<SPNAttachLog>(element.getLoc(), extractOp, extractOp.getType());
      }
      for (auto* escapingUser : escapingUsers.lookup(element)) {
        updateOperand(escapingUser, element, extractOp);
      }
      escapingUsers.erase(element);
    }
  }
}

Value ConversionManager::getValue(Superword* superword) const {
  return conversionState->getValue(superword);
}

Value ConversionManager::getOrCreateConstant(Location const& loc, Attribute const& attribute) {
  return folder.getOrCreateConstant(rewriter, &attribute.getDialect(), attribute, attribute.getType(), loc);
}

bool ConversionManager::hasEscapingUsers(Value const& value) const {
  return escapingUsers.count(value) && !escapingUsers.lookup(value).empty();
}

Value ConversionManager::getOrExtractValue(Value const& value) {
  if (conversionState->alreadyComputed(value)) {
    return value;
  }
  if (!costModel->isExtractionProfitable(value)) {
    conversionState->markComputed(value);
    return value;
  }
  auto const& wordPosition = conversionState->getWordContainingValue(value);
  assert(wordPosition.superword && "extraction deemed profitable, but value does not appear in any vector");
  auto const& source = conversionState->getValue(wordPosition.superword);
  auto const& pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) wordPosition.index));
  auto extractOp = rewriter.create<vector::ExtractElementOp>(value.getLoc(), source, pos);
  conversionState->markExtracted(value);
  return extractOp;
}

void ConversionManager::updateOperand(Operation* op, Value oldOperand, Value newOperand) {
  originalOperands.try_emplace(op, op->getOperands());
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    if (op->getOperand(i) == oldOperand) {
      op->setOperand(i, newOperand);
    }
  }
}
