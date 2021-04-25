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

bool NodeVector::isUniform() const {
  return std::all_of(std::begin(values), std::end(values), [&](Value const& value) {
    if (value.isa<BlockArgument>()) {
      return false;
    }
    return value.getDefiningOp()->getName() == values.front().getDefiningOp()->getName();
  });
}

bool NodeVector::contains(Value const& value) const {
  return std::find(std::begin(values), std::end(values), value) != std::end(values);
}

bool NodeVector::containsBlockArgs() const {
  return std::any_of(std::begin(values), std::end(values), [&](Value const& value) {
    return value.isa<BlockArgument>();
  });
}

bool NodeVector::vectorizable() const {
  return std::all_of(std::begin(values), std::end(values), [&](Value const& value) {
    return low::slp::vectorizable(value);
  });
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
  addVector(values, nullptr);
}

SLPNode::SLPNode(ArrayRef<Operation*> const& operations) {
  addVector(operations);
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

NodeVector* SLPNode::addVector(ArrayRef<Value> const& values, NodeVector* definingVector) {
  auto newVector = vectors.emplace_back(std::make_shared<NodeVector>(values));
  if (definingVector) {
    definingVector->operands.emplace_back(newVector);
  }
  return newVector.get();
}

NodeVector* SLPNode::addVector(ArrayRef<Operation*> const& operations) {
  return vectors.emplace_back(std::make_shared<NodeVector>(operations)).get();
}

NodeVector* SLPNode::getVector(size_t index) const {
  assert(index <= numVectors());
  return vectors[index].get();
}

SLPNode* SLPNode::addOperand(ArrayRef<Value> const& values, NodeVector* definingVector) {
  auto const& operandNode = operandNodes.emplace_back(std::make_unique<SLPNode>(values));
  if (definingVector) {
    definingVector->operands.emplace_back(operandNode->vectors.front());
  }
  return operandNode.get();
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
  for (auto* operand : root->getOperands()) {
    assert(std::find(std::begin(order), std::end(order), operand) == std::end(order));
    order.append(postOrder(operand));
  }
  order.emplace_back(root);
  return order;
}

/// Traverse the entire graph starting at root and gather lanes of each node vector that has escaping uses.
DenseMap<NodeVector*, std::shared_ptr<SmallPtrSetImpl<size_t>>> SLPNode::escapingLanesMap(SLPNode* root) {
  DenseMap<Value, unsigned> outsideUses;
  for (auto* node : postOrder(root)) {
    for (size_t i = node->numVectors(); i-- > 0;) {
      auto* vector = node->getVector(i);
      for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
        auto const& element = vector->getElement(lane);
        // Skip duplicate (splat) values.
        if (outsideUses.count(element)) {
          continue;
        }
        outsideUses[element] = std::distance(std::begin(element.getUses()), std::end(element.getUses()));
        for (size_t j = 0; j < vector->numOperands(); ++j) {
          auto* operand = vector->getOperand(j);
          assert(outsideUses[operand->getElement(lane)] > 0);
          outsideUses[operand->getElement(lane)]--;
        }
      }
    }
  }

  DenseMap<NodeVector*, std::shared_ptr<SmallPtrSetImpl<size_t>>> escapingLanes;
  for (auto* node : postOrder(root)) {
    for (size_t i = 0; i < node->numVectors(); ++i) {
      auto* vector = node->getVector(i);
      for (size_t lane = 0; lane < vector->numLanes(); ++lane) {
        if (outsideUses[vector->getElement(lane)] > 0) {
          if (escapingLanes.count(vector)) {
            escapingLanes[vector]->insert(lane);
          } else {
            SmallPtrSet<size_t, 4> lanes{lane};
            escapingLanes[vector] = std::make_shared<SmallPtrSet<size_t, 4>>(lanes);
          }
        }
      }
    }
  }
  return escapingLanes;
}
