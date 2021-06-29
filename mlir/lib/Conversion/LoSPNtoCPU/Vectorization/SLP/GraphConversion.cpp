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

ConversionState::ConversionState(std::shared_ptr<Superword> root, std::shared_ptr<ConversionState> parentState)
    : correspondingGraph{std::move(root)}, parentState{std::move(parentState)} {}

bool ConversionState::alreadyComputed(Superword* superword) const {
  if (computedSuperwords.count(superword)) {
    return true;
  }
  return parentState && parentState->alreadyComputed(superword);
}

bool ConversionState::alreadyComputed(Value value) const {
  return computedScalarValues.contains(value) || extractedScalarValues.contains(value);
}

bool ConversionState::isExtractable(Value value) {
  if (extractableScalarValues.count(value)) {
    return true;
  }
  return parentState && parentState->isExtractable(value);
}

void ConversionState::markComputed(Value value) {
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

void ConversionState::markExtracted(Value value) {
  if (extractedScalarValues.insert(value).second) {
    for (auto const& callback : extractionCallbacks) {
      callback(value);
    }
  }
}

Value ConversionState::getValue(Superword* superword) const {
  auto value = computedSuperwords.lookup(superword);
  if (value) {
    return value;
  } else if (parentState) {
    return parentState->getValue(superword);
  }
  llvm_unreachable("the superword has not yet been converted");
}

ValuePosition ConversionState::getSuperwordContainingValue(Value value) const {
  auto valuePosition = extractableScalarValues.lookup(value);
  if (!valuePosition && parentState) {
    return parentState->getSuperwordContainingValue(value);
  }
  return valuePosition;
}

SmallVector<Superword*> ConversionState::unconvertedPostOrder() const {
  SmallVector<Superword*> order;
  DenseMap<Superword*, unsigned> depths;
  depths[correspondingGraph.get()] = 0;
  SmallVector<Superword*> worklist{correspondingGraph.get()};
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

std::shared_ptr<ConversionState> ConversionState::getParentState() const {
  return parentState;
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

// === ConversionManager === //

ConversionManager::ConversionManager(PatternRewriter& rewriter, Block* block, std::shared_ptr<CostModel> costModel)
    : block{block}, trashBlock{nullptr}, costModel{std::move(costModel)},
      conversionState{std::make_shared<ConversionState>()}, rewriter{rewriter}, folder{rewriter.getContext()} {}

SmallVector<Superword*> ConversionManager::initConversion(SLPGraph const& graph) {
  // Clear all temporary data.
  escapingUsers.clear();
  originalOperations.clear();
  originalOperands.clear();
  // Work on a new, temporary conversion state.
  conversionState = std::make_shared<ConversionState>(graph.getRoot(), conversionState);
  costModel->setConversionState(conversionState);
  // Store original block state for undoing graph conversions.
  block->walk([&](Operation* op) {
    originalOperations.emplace_back(op);
    originalOperands[op] = op->getOperands();
  });
  // Gather escaping users.
  auto order = conversionState->unconvertedPostOrder();
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
        if (!superword->isLeaf()) {
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
      return lhs->isBeforeInBlock(rhs);
    });
  }
  return order;
}

void ConversionManager::finishConversion() {
  // Sort operations topologically.
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
  // Perform DCE on the resulting block. This is required for correct cost analyses.
  SmallPtrSet<Operation*, 32> deadOps;
  block->walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isOpTriviallyDead(op)) {
      worklist.insert(op);
      deadOps.insert(op);
    }
  });
  while (!worklist.empty()) {
    auto* op = worklist.pop_back_val();
    for (auto const& operand : op->getOperands()) {
      if (auto* operandOp = operand.getDefiningOp()) {
        auto users = operandOp->getUsers();
        if (std::all_of(std::begin(users), std::end(users), [&](Operation* user) {
          return deadOps.contains(user);
        })) {
          worklist.insert(operandOp);
          deadOps.insert(operandOp);
        }
      }
    }
  }
  moveToTrash(deadOps);
}

void ConversionManager::acceptConversion() {
  emptyTrash();
}

void ConversionManager::rejectConversion() {
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
  conversionState->undoChanges();
  conversionState = conversionState->getParentState();
  // Move all created vector operations to a trash block, so that they can be destroyed *immediately* instead of at the
  // end of the conversion process. We don't want to carry thousands or millions of erased operations with us to the
  // next vectorization attempt.
  moveToTrash(erasableOps);
  emptyTrash();
}

void ConversionManager::setupConversionFor(Superword* superword, SLPVectorizationPattern const* pattern) {
  rewriter.setInsertionPointToEnd(block);
  // Create extractions if needed.
  auto scalarInputs = leafVisitor.getRequiredScalarValues(pattern, superword);
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    auto const& element = superword->getElement(lane);
    if (std::find(std::begin(scalarInputs), std::end(scalarInputs), element) != std::end(scalarInputs)) {
      superword->setElement(lane, getOrExtractValue(element));
    }
  }
}

void ConversionManager::update(Superword* superword, Value operation, SLPVectorizationPattern const* appliedPattern) {
  if (conversionState->alreadyComputed(superword)) {
    dumpSuperword(*superword);
    getValue(superword).dump();
  }
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
                logExtractOp = rewriter.create<SPNAttachLog>(element.getLoc(), extractOp, extractOp.getType());
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
  auto const& source = conversionState->getValue(wordPosition.superword);
  auto const& pos = getOrCreateConstant(source.getLoc(), rewriter.getI32IntegerAttr((int) wordPosition.index));
  auto extractOp = rewriter.create<vector::ExtractElementOp>(value.getLoc(), source, pos);
  conversionState->markExtracted(value);
  return extractOp;
}

void ConversionManager::moveToTrash(SmallPtrSetImpl<Operation*> const& deadOps) {
  if (!trashBlock) {
    trashBlock = rewriter.createBlock(block);
    block->moveBefore(trashBlock);
  }
  for (auto* op : deadOps) {
    //op->dropAllReferences();
    op->dropAllUses();
    if (op->getNumOperands() > 0) {
      op->eraseOperands(0, op->getNumOperands());
    }
    op->moveBefore(trashBlock, trashBlock->end());
  }
}

void ConversionManager::emptyTrash() {
  trashBlock->walk([&](Operation* op) {
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      folder.notifyRemoval(op);
    }
  });
  rewriter.eraseBlock(trashBlock);
  trashBlock = nullptr;
}
