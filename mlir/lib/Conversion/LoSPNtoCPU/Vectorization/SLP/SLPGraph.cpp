//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/SLPGraph.h"
#include "LoSPNtoCPU/Vectorization/SLP/Util.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// === NodeVector === //

ValueVector::ValueVector(ArrayRef<Value> const& values) {
  assert(!values.empty());
  for (auto const& value : values) {
    assert(value.isa<BlockArgument>() || value.getDefiningOp()->hasTrait<OpTrait::OneResult>());
    this->values.emplace_back(value);
  }
}

ValueVector::ValueVector(ArrayRef<Operation*> const& operations) {
  assert(!operations.empty());
  for (auto* op : operations) {
    assert(op->hasTrait<OpTrait::OneResult>());
    values.emplace_back(op->getResult(0));
  }
}

Value ValueVector::getElement(size_t lane) const {
  assert(lane < numLanes());
  return values[lane];
}

void ValueVector::setElement(size_t lane, Value const& value) {
  assert(lane < numLanes());
  values[lane] = value;
}

Value ValueVector::operator[](size_t lane) const {
  return getElement(lane);
}

bool ValueVector::contains(Value const& value) const {
  return std::find(std::begin(values), std::end(values), value) != std::end(values);
}

bool ValueVector::isLeaf() const {
  return operandVectors.empty();
}

bool ValueVector::uniform() const {
  Operation* firstOp = nullptr;
  for (size_t i = 0; i < values.size(); ++i) {
    if (auto* definingOp = values[i].getDefiningOp()) {
      if (i == 0) {
        firstOp = definingOp;
        continue;
      }
      if (firstOp->getName() != definingOp->getName()) {
        return false;
      }
    } else if (firstOp) {
      return false;
    }
  }
  return true;
}

bool ValueVector::splattable() const {
  Operation* firstOp = nullptr;
  for (size_t i = 0; i < values.size(); ++i) {
    if (auto* definingOp = values[i].getDefiningOp()) {
      if (i == 0) {
        firstOp = definingOp;
        continue;
      }
      if (!OperationEquivalence::isEquivalentTo(definingOp, firstOp)) {
        return false;
      }
    } else if (firstOp || values[i] != values.front()) {
      return false;
    }
  }
  return true;
}

size_t ValueVector::numLanes() const {
  return values.size();
}

SmallVectorImpl<Value>::const_iterator ValueVector::begin() const {
  return values.begin();
}

SmallVectorImpl<Value>::const_iterator ValueVector::end() const {
  return values.end();
}

size_t ValueVector::numOperands() const {
  return operandVectors.size();
}

void ValueVector::addOperand(std::shared_ptr<ValueVector> operandVector) {
  operandVectors.emplace_back(std::move(operandVector));
}

ValueVector* ValueVector::getOperand(size_t index) const {
  assert(index < operandVectors.size());
  return operandVectors[index].get();
}

SmallVector<ValueVector*, 2> ValueVector::getOperands() const {
  SmallVector<ValueVector*, 2> operands;
  for (auto const& operand : operandVectors) {
    operands.emplace_back(operand.get());
  }
  return operands;
}

// === SLPNode === //

SLPNode::SLPNode(std::shared_ptr<ValueVector> vector) {
  vectors.emplace_back(std::move(vector));
}

void SLPNode::addVector(std::shared_ptr<ValueVector> vector) {
  vectors.emplace_back(std::move(vector));
}

std::shared_ptr<ValueVector> SLPNode::getVector(size_t index) const {
  assert(index <= numVectors());
  return vectors[index];
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

ArrayRef<std::shared_ptr<SLPNode>> SLPNode::getOperands() const {
  return operandNodes;
}
