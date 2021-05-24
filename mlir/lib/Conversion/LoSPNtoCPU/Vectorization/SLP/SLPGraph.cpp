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

// === Superword === //

Superword::Superword(ArrayRef<Value> const& values) {
  assert(!values.empty());
  for (auto const& value : values) {
    assert(value.isa<BlockArgument>() || value.getDefiningOp()->hasTrait<OpTrait::OneResult>());
    this->values.emplace_back(value);
  }
}

Superword::Superword(ArrayRef<Operation*> const& operations) {
  assert(!operations.empty());
  for (auto* op : operations) {
    assert(op->hasTrait<OpTrait::OneResult>());
    values.emplace_back(op->getResult(0));
  }
}

Value Superword::getElement(size_t lane) const {
  assert(lane < numLanes());
  return values[lane];
}

void Superword::setElement(size_t lane, Value const& value) {
  assert(lane < numLanes());
  values[lane] = value;
}

Value Superword::operator[](size_t lane) const {
  return getElement(lane);
}

bool Superword::contains(Value const& value) const {
  return std::find(std::begin(values), std::end(values), value) != std::end(values);
}

bool Superword::isLeaf() const {
  return operandWords.empty();
}

bool Superword::uniform() const {
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

bool Superword::splattable() const {
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

size_t Superword::numLanes() const {
  return values.size();
}

SmallVectorImpl<Value>::const_iterator Superword::begin() const {
  return values.begin();
}

SmallVectorImpl<Value>::const_iterator Superword::end() const {
  return values.end();
}

size_t Superword::numOperands() const {
  return operandWords.size();
}

void Superword::addOperand(std::shared_ptr<Superword> operandWord) {
  operandWords.emplace_back(std::move(operandWord));
}

Superword* Superword::getOperand(size_t index) const {
  assert(index < operandWords.size());
  return operandWords[index].get();
}

SmallVector<Superword*, 2> Superword::getOperands() const {
  SmallVector<Superword*, 2> operands;
  for (auto const& operand : operandWords) {
    operands.emplace_back(operand.get());
  }
  return operands;
}

VectorType Superword::getVectorType() const {
  return VectorType::get(static_cast<unsigned>(numLanes()), getElementType());
}

Type Superword::getElementType() const {
  return getElement(0).getType();
}

Location Superword::getLoc() const {
  return getElement(0).getLoc();
}

// === SLPNode === //

SLPNode::SLPNode(std::shared_ptr<Superword> superword) {
  superwords.emplace_back(std::move(superword));
}

void SLPNode::addSuperword(std::shared_ptr<Superword> superword) {
  superwords.emplace_back(std::move(superword));
}

std::shared_ptr<Superword> SLPNode::getSuperword(size_t index) const {
  assert(index <= numSuperwords());
  return superwords[index];
}

Value SLPNode::getValue(size_t lane, size_t index) const {
  assert(lane <= numLanes() && index <= numSuperwords());
  return superwords[index]->values[lane];
}

void SLPNode::setValue(size_t lane, size_t index, Value const& newValue) {
  assert(lane <= numLanes() && index <= numSuperwords());
  superwords[index]->values[lane] = newValue;
}

bool SLPNode::contains(Value const& value) const {
  return std::any_of(std::begin(superwords), std::end(superwords), [&](auto const& superword) {
    return superword->contains(value);
  });
}

bool SLPNode::isSuperwordRoot(Superword const& superword) const {
  return superwords[0]->values == superword.values;
}

size_t SLPNode::numLanes() const {
  return superwords[0]->numLanes();
}

size_t SLPNode::numSuperwords() const {
  return superwords.size();
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
