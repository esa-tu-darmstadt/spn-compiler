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

using namespace mlir;
using namespace mlir::spn::low::slp;

// === ConversionState === //

void ConversionState::startConversion(std::shared_ptr<Superword> root) {
  correspondingGraphs.emplace_back(std::move(root));
}

void ConversionState::finishConversion() {
  permanentData.mergeWith(temporaryData);
  temporaryData.clear();
}

void ConversionState::cancelConversion() {
  for (auto const& callback : scalarUndoCallbacks) {
    for (auto value : temporaryData.computedScalarValues) {
      if (permanentData.alreadyComputed(value)) {
        continue;
      }
      callback(value);
    }
  }
  for (auto const& callback : vectorUndoCallbacks) {
    for (auto const& entry : temporaryData.computedSuperwords) {
      callback(entry.first);
    }
  }
  for (auto const& callback : extractionUndoCallbacks) {
    for (auto value : temporaryData.extractedScalarValues) {
      if (permanentData.alreadyComputed(value)) {
        continue;
      }
      callback(value);
    }
  }
  temporaryData.clear();
  correspondingGraphs.pop_back();
}

bool ConversionState::alreadyComputed(Superword* superword) const {
  if (temporaryData.computedSuperwords.count(superword)) {
    return true;
  }
  return permanentData.computedSuperwords.count(superword);
}

bool ConversionState::alreadyComputed(Value value) const {
  return permanentData.alreadyComputed(value) || temporaryData.alreadyComputed(value);
}

bool ConversionState::isExtractable(Value value) {
  if (temporaryData.extractableScalarValues.count(value)) {
    return true;
  }
  return permanentData.extractableScalarValues.count(value);
}

void ConversionState::markComputed(Value value) {
  assert(!alreadyComputed(value) && "marking already computed value as computed");
  if (temporaryData.computedScalarValues.insert(value).second) {
    for (auto const& callback : scalarCallbacks) {
      callback(value);
    }
    if (auto* definingOp = value.getDefiningOp()) {
      for (auto const& operand : definingOp->getOperands()) {
        if (!alreadyComputed(operand)) {
          markComputed(operand);
        }
      }
    }
  }
}

void ConversionState::markComputed(Superword* superword, Value value) {
  assert(!alreadyComputed(superword) && "superword has already been converted");
  for (auto* operand : superword->getOperands()) {
    assert(alreadyComputed(operand) && "computing vector before its operands");
  }
  temporaryData.computedSuperwords[superword] = value;
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    temporaryData.extractableScalarValues.try_emplace(superword->getElement(lane), superword, lane);
  }
  for (auto const& callback : vectorCallbacks) {
    callback(superword);
  }
}

void ConversionState::markExtracted(Value value) {
  assert(!alreadyComputed(value) && "extracting value that has been marked as computed already");
  if (temporaryData.extractedScalarValues.insert(value).second) {
    for (auto const& callback : extractionCallbacks) {
      callback(value);
    }
  }
}

Value ConversionState::getValue(Superword* superword) const {
  auto value = temporaryData.computedSuperwords.lookup(superword);
  if (value) {
    return value;
  }
  value = permanentData.computedSuperwords.lookup(superword);
  assert(value && "the superword has not yet been converted");
  return value;
}

ValuePosition ConversionState::getSuperwordContainingValue(Value value) const {
  auto valuePosition = temporaryData.extractableScalarValues.lookup(value);
  if (valuePosition) {
    return valuePosition;
  }
  valuePosition = permanentData.extractableScalarValues.lookup(value);
  assert(valuePosition && "there is no superword containing the value");
  return valuePosition;
}

SmallVector<Superword*> ConversionState::unconvertedPostOrder() const {
  SmallVector<Superword*> order;
  DenseMap<Superword*, unsigned> depths;
  depths[correspondingGraphs.back().get()] = 0;
  SmallVector<Superword*> worklist{correspondingGraphs.back().get()};
  while (!worklist.empty()) {
    auto* superword = worklist.pop_back_val();
    if (alreadyComputed(superword)) {
      continue;
    }
    for (auto* operand : superword->getOperands()) {
      auto operandDepth = depths[superword] + 1;
      if (depths[operand] < operandDepth) {
        depths[operand] = operandDepth;
        worklist.emplace_back(operand);
      }
    }
  }
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

void ConversionState::addVectorCallbacks(std::function<void(Superword*)> createCallback,
                                         std::function<void(Superword*)> undoCallback) {
  vectorCallbacks.emplace_back(std::move(createCallback));
  vectorUndoCallbacks.emplace_back(std::move(undoCallback));
}

void ConversionState::addScalarCallbacks(std::function<void(Value)> inputCallback,
                                         std::function<void(Value)> undoCallback) {
  scalarCallbacks.emplace_back(std::move(inputCallback));
  scalarUndoCallbacks.emplace_back(std::move(undoCallback));
}

void ConversionState::addExtractionCallbacks(std::function<void(Value)> extractCallback,
                                             std::function<void(Value)> undoCallback) {
  extractionCallbacks.emplace_back(std::move(extractCallback));
  extractionUndoCallbacks.emplace_back(std::move(undoCallback));
}

// === ConversionManager === //

ConversionManager::ConversionManager(RewriterBase& rewriter, Block* block, std::shared_ptr<CostModel> costModel)
    : block{block}, costModel{std::move(costModel)}, conversionState{std::make_shared<ConversionState>()},
      rewriter{rewriter}, folder{rewriter.getContext()} {
  this->costModel->setConversionState(conversionState);
}

SmallVector<Superword*> ConversionManager::startConversion(SLPGraph const& graph) {
  // Clear all temporary data.
  escapingUsers.clear();
  originalOperations.clear();
  originalOperands.clear();
  // Work on a new, temporary conversion state.
  conversionState->startConversion(graph.getRoot());
  // Store original block state for undoing graph conversions.
  block->walk([&](Operation* op) {
    originalOperations.emplace_back(op);
    originalOperands[op] = op->getOperands();
  });
  // Gather escaping users.
  auto order = conversionState->unconvertedPostOrder();
  for (auto* superword : order) {
    for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
      auto element = superword->getElement(lane);
      if (auto* elementOp = element.getDefiningOp()) {
        if (elementOp->hasTrait<OpTrait::ConstantLike>()) {
          continue;
        }
        if (!escapingUsers.count(element)) {
          escapingUsers[element].assign(std::begin(element.getUsers()), std::end(element.getUsers()));
        }
        if (!superword->isLeaf()) {
          for (size_t i = 0; i < superword->numOperands(); ++i) {
            auto operand = superword->getOperand(i)->getElement(lane);
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
      return lhs->isBeforeInBlock(rhs);
    });
  }
  return order;
}

void ConversionManager::finishConversion() {
  reorderOperations();
  conversionState->finishConversion();
}

void ConversionManager::cancelConversion() {
  Operation* lastOp = nullptr;
  for (auto* op : originalOperations) {
    auto operands = originalOperands.lookup(op);
    if (!operands.empty()) {
      op->setOperands(operands);
    }
    if (lastOp) {
      op->moveAfter(lastOp);
    } else {
      op->moveBefore(block, block->begin());
    }
    lastOp = op;
  }
  SmallPtrSet<Operation*, 32> erasableOps;
  // Every op that appears after the last 'original' op can be erased.
  while (auto* trashOp = lastOp->getNextNode()) {
    erasableOps.insert(trashOp);
    lastOp = trashOp;
  }
  conversionState->cancelConversion();
  for (auto* op : erasableOps) {
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      folder.notifyRemoval(op);
    }
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
}

void ConversionManager::setupConversionFor(Superword* superword, SLPVectorizationPattern const* pattern) {
  rewriter.setInsertionPoint(block->getTerminator());
  // Create extractions if needed.
  auto scalarInputs = leafVisitor.getRequiredScalarValues(pattern, superword);
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    auto element = superword->getElement(lane);
    if (std::find(std::begin(scalarInputs), std::end(scalarInputs), element) != std::end(scalarInputs)) {
      superword->setElement(lane, getOrExtractValue(element));
    }
  }
}

void ConversionManager::update(Superword* superword, Value operation, SLPVectorizationPattern const* appliedPattern) {
  conversionState->markComputed(superword, operation);
  auto scalarInputs = leafVisitor.getRequiredScalarValues(appliedPattern, superword);
  for (auto scalarInput : scalarInputs) {
    if (!conversionState->alreadyComputed(scalarInput)) {
      conversionState->markComputed(scalarInput);
    }
  }
  // Create vector extractions for escaping uses.
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    auto element = superword->getElement(lane);
    if (conversionState->alreadyComputed(element)) {
      continue;
    }
    if (hasEscapingUsers(element)) {
      Value extractOp = getOrExtractValue(element);
      // Nothing to do if it's being computed in scalar form somewhere else.
      if (extractOp != element) {
        Value logExtractOp = nullptr;
        for (auto* escapingUser : escapingUsers.lookup(element)) {
          for (size_t i = 0; i < escapingUser->getNumOperands(); ++i) {
            if (escapingUser->getOperand(i) != element) {
              continue;
            }
            if (escapingUser->getOperand(i).getType().isa<LogType>()) {
              if (!logExtractOp) {
                logExtractOp = rewriter.create<SPNConvertLog>(element.getLoc(), extractOp);
              }
              escapingUser->setOperand(i, logExtractOp);
            } else {
              escapingUser->setOperand(i, extractOp);
            }
          }
        }
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

ConversionState& ConversionManager::getConversionState() const {
  return *conversionState;
}

bool ConversionManager::hasEscapingUsers(Value value) const {
  return escapingUsers.count(value) && !escapingUsers.lookup(value).empty();
}

Value ConversionManager::getOrExtractValue(Value value) {
  if (conversionState->alreadyComputed(value)) {
    return value;
  }
  if (!costModel->isExtractionProfitable(value)) {
    conversionState->markComputed(value);
    return value;
  }
  auto const& wordPosition = conversionState->getSuperwordContainingValue(value);
  assert(wordPosition.superword && "extraction deemed profitable, but value does not appear in any vector");
  auto source = conversionState->getValue(wordPosition.superword);
  auto pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) wordPosition.index));
  auto extractOp = rewriter.create<vector::ExtractElementOp>(value.getLoc(), source, pos);
  conversionState->markExtracted(value);
  return extractOp;
}

// Helper functions in anonymous namespace.
namespace {
  void reorderOperationsBFS(Block* block) {
    DenseMap<Operation*, unsigned> depths;
    llvm::SmallSetVector<Operation*, 32> worklist;
    block->walk<WalkOrder::PreOrder>([&](Operation* op) {
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
          op->moveBefore(block, block->begin());
        }
        lastOp = op;
      }
    }
  }

  void reorderOperationsDFS(Block* block) {
    llvm::SmallSetVector<Operation*, 128> order;
    for (auto it = block->rbegin(); it != block->rend(); ++it) {
      auto* op = &(*it);
      if (order.contains(op) || op->getNumOperands() == 0 || op == block->getTerminator()) {
        continue;
      }
      // true: all operands done, can be inserted into order
      // false: need to visit operands
      SmallVector<std::pair<Operation*, bool>> stack;
      stack.emplace_back(op, false);
      while (!stack.empty()) {
        auto pair = stack.pop_back_val();
        if (order.contains(pair.first) || pair.first->getNumOperands() == 0) {
          continue;
        }
        if (pair.second) {
          order.insert(pair.first);
          continue;
        }
        stack.emplace_back(pair.first, true);
        // Reverse order due to LIFO stack. Otherwise, RHS operands would appear before LHS operands in the final order.
        for (unsigned i = pair.first->getNumOperands(); i-- > 0;) {
          auto operand = pair.first->getOperand(i);
          if (auto* definingOp = operand.getDefiningOp()) {
            // Skip finished operations.
            if (order.contains(definingOp) || definingOp->getNumOperands() == 0) {
              continue;
            }
            stack.emplace_back(definingOp, false);
          }
        }
      }
    }
    Operation* lastOp = nullptr;
    for (auto* op : order) {
      if (lastOp) {
        op->moveAfter(lastOp);
      } else {
        op->moveBefore(block->getTerminator());
      }
      lastOp = op;
    }
  }
}

void ConversionManager::reorderOperations() {
  reorderOperationsDFS(block);
}
