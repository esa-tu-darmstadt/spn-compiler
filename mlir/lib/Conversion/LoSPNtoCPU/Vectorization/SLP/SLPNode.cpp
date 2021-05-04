//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPNode.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// === NodeVector === //

NodeVector::NodeVector(ArrayRef<Value> const& values) {
  assert(!values.empty());
  for (auto const& value : values) {
    assert(value.isa<BlockArgument>() || value.getDefiningOp()->hasTrait<OpTrait::OneResult>());
    this->values.emplace_back(value);
  }
}

NodeVector::NodeVector(ArrayRef<Operation*> const& operations) {
  assert(!operations.empty());
  for (auto* op : operations) {
    assert(op->hasTrait<OpTrait::OneResult>());
    values.emplace_back(op->getResult(0));
  }
}

bool NodeVector::contains(Value const& value) const {
  return std::find(std::begin(values), std::end(values), value) != std::end(values);
}

bool NodeVector::containsBlockArgs() const {
  return std::any_of(std::begin(values), std::end(values), [&](Value const& value) {
    return value.isa<BlockArgument>();
  });
}

bool NodeVector::splattable() const {
  if (containsBlockArgs()) {
    return std::all_of(std::begin(values), std::end(values), [&](Value const& element) {
      return element == values.front();
    });
  }
  return std::all_of(std::begin(values), std::end(values), [&](Value const& element) {
    return OperationEquivalence::isEquivalentTo(element.getDefiningOp(), values.front().getDefiningOp());
  });
}

bool NodeVector::isLeaf() const {
  return operands.empty();
}

size_t NodeVector::numLanes() const {
  return values.size();
}

size_t NodeVector::numOperands() const {
  return operands.size();
}

NodeVector* NodeVector::getOperand(size_t index) const {
  assert(index < operands.size());
  return operands[index];
}

SmallVectorImpl<Value>::const_iterator NodeVector::begin() const {
  return values.begin();
}

SmallVectorImpl<Value>::const_iterator NodeVector::end() const {
  return values.end();
}

Value const& NodeVector::getElement(size_t lane) const {
  return this->operator[](lane);
}

Value const& NodeVector::operator[](size_t lane) const {
  assert(lane < numLanes());
  return values[lane];
}

// === SLPNode === //

SLPNode::SLPNode(ArrayRef<Value> const& values) {
  vectors.emplace_back(std::make_unique<NodeVector>(values));
}

SLPNode::SLPNode(ArrayRef<Operation*> const& operations) {
  vectors.emplace_back(std::make_unique<NodeVector>(operations));
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

bool SLPNode::isRootOfNode(NodeVector const& vector) const {
  return vectors[0]->values == vector.values;
}

size_t SLPNode::numLanes() const {
  return vectors[0]->numLanes();
}

size_t SLPNode::numVectors() const {
  return vectors.size();
}

NodeVector* SLPNode::addVector(ArrayRef<Value> const& values, NodeVector* definingVector) {
  auto const& newVector = vectors.emplace_back(std::make_unique<NodeVector>(values));
  definingVector->operands.emplace_back(newVector.get());
  return newVector.get();
}

NodeVector* SLPNode::getVector(size_t index) const {
  assert(index <= numVectors());
  return vectors[index].get();
}

NodeVector* SLPNode::getVectorOrNull(ArrayRef<Value> const& values) const {
  auto it = std::find_if(std::begin(vectors), std::end(vectors), [&](auto const& vector) {
    return values.equals(vector->values);
  });
  return it != std::end(vectors) ? it->get() : nullptr;
}

void SLPNode::addOperand(std::shared_ptr<SLPNode> operandNode, NodeVector* operandVector, NodeVector* definingVector) {
  operandNodes.emplace_back(std::move(operandNode));
  definingVector->operands.emplace_back(operandVector);
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

size_t SLPNode::numOperands() const {
  return operandNodes.size();
}

// === Utilities === //

SmallVector<SLPNode*> SLPNode::postOrder(SLPNode* root) {
  SmallVector<SLPNode*> order;
  // false = visit operands, true = insert into order
  std::vector<std::pair<SLPNode*, bool>> worklist;
  worklist.emplace_back(root, false);
  while (!worklist.empty()) {
    auto* node = worklist.back().first;
    bool insert = worklist.back().second;
    worklist.pop_back();
    if (insert) {
      order.emplace_back(node);
    } else {
      worklist.emplace_back(node, true);
      for (auto* operand: node->getOperands()) {
        if (std::find(std::begin(order), std::end(order), operand) == std::end(order)) {
          worklist.emplace_back(operand, false);
        }
      }
    }
  }
  return order;
}
