//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPNode.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// === NodeVector ===

NodeVector::NodeVector(vector_t const& values) {
  assert(!values.empty());
  for (auto const& value : values) {
    assert(value.isa<BlockArgument>() || value.getDefiningOp()->hasTrait<OpTrait::OneResult>());
    this->values.emplace_back(value);
  }
}

NodeVector::NodeVector(SmallVector<Operation*, 4> const& operations) {
  assert(!operations.empty());
  for (auto* op : operations) {
    assert(op->hasTrait<OpTrait::OneResult>());
    values.emplace_back(op->getResult(0));
  }
}

bool NodeVector::isUniform() const {
  return std::all_of(std::begin(values), std::end(values), [&](Value const& value) {
    if (value.isa<BlockArgument>()) {
      return false;
    }
    return value.getDefiningOp()->getName() == values.front().getDefiningOp()->getName();
  });
}

bool NodeVector::containsBlockArg() const {
  return std::any_of(std::begin(values), std::end(values), [&](Value const& value) {
    return value.isa<BlockArgument>();
  });
}

bool NodeVector::contains(Value const& value) const {
  return std::find(std::begin(values), std::end(values), value) != std::end(values);
}

size_t NodeVector::numLanes() const {
  return values.size();
}

size_t NodeVector::numOperands() const {
  return operands.size();
}

NodeVector* NodeVector::getOperand(size_t index) const {
  assert(index < operands.size());
  return operands[index].get();
}

vector_t::const_iterator NodeVector::begin() const {
  return values.begin();
}

vector_t::const_iterator NodeVector::end() const {
  return values.end();
}

Value const& NodeVector::getElement(size_t lane) const {
  return this->operator[](lane);
}

Value const& NodeVector::operator[](size_t lane) const {
  assert(lane < numLanes());
  return values[lane];
}

// === SLPNode ===

SLPNode::SLPNode(SmallVector<Operation*, 4> const& operations) {
  addVector(operations);
}

SLPNode::SLPNode(vector_t const& values) {
  addVector(values, nullptr);
}

Value SLPNode::getValue(size_t lane, size_t index) const {
  assert(lane <= numLanes() && index <= numVectors());
  return vectors[index]->values[lane];
}

void SLPNode::setValue(size_t lane, size_t index, Value const& newValue) {
  assert(lane <= numLanes() && index <= numVectors());
  vectors[index]->values[lane] = newValue;
}

bool SLPNode::isUniform() const {
  return std::all_of(std::begin(vectors), std::end(vectors), [&](auto const& nodeVector) {
    return nodeVector->isUniform();
  });
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

NodeVector* SLPNode::addVector(vector_t const& values, NodeVector* definingVector) {
  auto* newVector = vectors.emplace_back(std::make_shared<NodeVector>(values)).get();
  if (definingVector) {
    definingVector->operands.emplace_back(newVector);
  }
  return newVector;
}

NodeVector* SLPNode::addVector(SmallVector<Operation*, 4> const& operations) {
  return vectors.emplace_back(std::make_shared<NodeVector>(operations)).get();
}

NodeVector* SLPNode::getVector(size_t index) const {
  assert(index <= numVectors());
  return vectors[index].get();
}

SLPNode* SLPNode::addOperand(vector_t const& values, NodeVector* definingVector) {
  auto* operandNode = operandNodes.emplace_back(std::make_unique<SLPNode>(values)).get();
  if (definingVector) {
    definingVector->operands.emplace_back(operandNode->vectors.front());
  }
  return operandNode;
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
