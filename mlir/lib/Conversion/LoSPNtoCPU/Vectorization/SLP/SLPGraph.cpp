//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPGraph.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// === NodeVector === //

ValueVector::ValueVector(ArrayRef<Value> const& values, std::shared_ptr<SLPNode> const& parentNode) : parentNode{
    parentNode} {
  assert(!values.empty());
  for (auto const& value : values) {
    assert(value.isa<BlockArgument>() || value.getDefiningOp()->hasTrait<OpTrait::OneResult>());
    this->values.emplace_back(value);
  }
}

ValueVector::ValueVector(ArrayRef<Operation*> const& operations, std::shared_ptr<SLPNode> const& parentNode)
    : parentNode{parentNode} {
  assert(!operations.empty());
  for (auto* op : operations) {
    assert(op->hasTrait<OpTrait::OneResult>());
    values.emplace_back(op->getResult(0));
  }
}

bool ValueVector::contains(Value const& value) const {
  return std::find(std::begin(values), std::end(values), value) != std::end(values);
}

bool ValueVector::containsBlockArgs() const {
  return std::any_of(std::begin(values), std::end(values), [&](Value const& value) {
    return value.isa<BlockArgument>();
  });
}

bool ValueVector::splattable() const {
  if (containsBlockArgs()) {
    return std::all_of(std::begin(values), std::end(values), [&](Value const& element) {
      return element == values.front();
    });
  }
  return std::all_of(std::begin(values), std::end(values), [&](Value const& element) {
    return OperationEquivalence::isEquivalentTo(element.getDefiningOp(), values.front().getDefiningOp());
  });
}

bool ValueVector::isLeaf() const {
  return operands.empty();
}

size_t ValueVector::numLanes() const {
  return values.size();
}

size_t ValueVector::numOperands() const {
  return operands.size();
}

void ValueVector::addOperand(ValueVector* operandVector) {
  operands.emplace_back(operandVector);
}

ValueVector* ValueVector::getOperand(size_t index) const {
  assert(index < operands.size());
  return operands[index];
}

std::shared_ptr<SLPNode> ValueVector::getParentNode() const {
  if (auto parentPtr = parentNode.lock()) {
    return parentPtr;
  }
  assert(false && "parent node has expired already");
}

SmallVectorImpl<Value>::const_iterator ValueVector::begin() const {
  return values.begin();
}

SmallVectorImpl<Value>::const_iterator ValueVector::end() const {
  return values.end();
}

Value ValueVector::getElement(size_t lane) const {
  return this->operator[](lane);
}

Value ValueVector::operator[](size_t lane) const {
  assert(lane < numLanes());
  return values[lane];
}

// === SLPNode === //

ValueVector* SLPNode::addVector(std::unique_ptr<ValueVector> vector) {
  return vectors.emplace_back(std::move(vector)).get();
}

ValueVector* SLPNode::getVector(size_t index) const {
  assert(index <= numVectors());
  return vectors[index].get();
}

Value SLPNode::getValue(size_t lane, size_t index) const {
  assert(lane <= numLanes() && index <= numVectors());
  return vectors[index]->values[lane];
}

void SLPNode::setValue(size_t lane, size_t index, Value const& newValue) {
  assert(lane <= numLanes() && index <= numVectors());
  vectors[index]->values[lane] = newValue;
}

bool SLPNode::contains(Value const& value) const {
  return std::any_of(std::begin(vectors), std::end(vectors), [&](auto const& nodeVector) {
    return nodeVector->contains(value);
  });
}

bool SLPNode::isVectorRoot(ValueVector const& vector) const {
  return vectors[0]->values == vector.values;
}

size_t SLPNode::numLanes() const {
  return vectors[0]->numLanes();
}

size_t SLPNode::numVectors() const {
  return vectors.size();
}

size_t SLPNode::numOperands() const {
  return operandNodes.size();
}

void SLPNode::addOperand(std::shared_ptr<SLPNode> operandNode) {
  operandNodes.emplace_back(std::move(operandNode));
}

SLPNode* SLPNode::getOperand(size_t index) const {
  assert(index <= operandNodes.size());
  return operandNodes[index].get();
}

std::vector<SLPNode*> SLPNode::getOperands() const {
  std::vector<SLPNode*> operands;
  operands.reserve(operands.size());
  for (auto const& operand : operandNodes) {
    operands.emplace_back(operand.get());
  }
  return operands;
}

// === Utilities === //

SmallVector<SLPNode*> graph::postOrder(SLPNode* root) {
  SmallVector<SLPNode*> order;
  // false = visit operands, true = insert into order
  SmallVector<std::pair<SLPNode*, bool>> worklist;
  llvm::SmallSet<SLPNode*, 32> finishedNodes;
  worklist.emplace_back(root, false);
  while (!worklist.empty()) {
    if (finishedNodes.contains(worklist.back().first)) {
      worklist.pop_back();
      continue;
    }
    auto* node = worklist.back().first;
    bool insert = worklist.back().second;
    worklist.pop_back();
    if (insert) {
      order.emplace_back(node);
      finishedNodes.insert(node);
    } else {
      worklist.emplace_back(node, true);
      for (auto* operand: node->getOperands()) {
        worklist.emplace_back(operand, false);
      }
    }
  }
  return order;
}
