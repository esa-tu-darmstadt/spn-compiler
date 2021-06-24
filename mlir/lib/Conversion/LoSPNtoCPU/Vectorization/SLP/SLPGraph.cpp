//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/SLPGraph.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPN/LoSPNTypes.h"

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

bool Superword::constant() const {
  for (auto const& value : values) {
    if (auto* definingOp = value.getDefiningOp()) {
      if (!definingOp->hasTrait<OpTrait::ConstantLike>()) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
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
  if (auto logType = getElement(0).getType().dyn_cast<LogType>()) {
    return VectorType::get(static_cast<unsigned>(numLanes()), logType.getBaseType());
  }
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

// === SLPGraph === //

SLPGraph::SLPGraph(ArrayRef<Value> const& seed, unsigned lookAhead) : lookAhead{lookAhead} {
  SLPGraphBuilder builder{*this};
  builder.build(seed);
}

std::shared_ptr<Superword> SLPGraph::getRoot() const {
  return root;
}

DependencyGraph SLPGraph::dependencyGraph() const {
  DependencyGraph dependencyGraph;
  DenseMap<Value, SmallPtrSet<Superword*, 2>> valueOccurrences;
  graph::walk(root.get(), [&](Superword* superword) {
    for (auto const& element : *superword) {
      valueOccurrences[element].insert(superword);
    }
    dependencyGraph.nodes.insert(superword);
  });
  // BFS through scalar values.
  llvm::SmallSetVector<Value, 32> worklist;
  llvm::SmallSetVector<Value, 32> nextWorklist;
  DenseMap<Value, SmallPtrSet<Superword*, 32>> reachableUses;
  for (auto const& element : *root) {
    if (auto* definingOp = element.getDefiningOp()) {
      for (auto const& operand : definingOp->getOperands()) {
        reachableUses[operand].insert(root.get());
        nextWorklist.insert(operand);
      }
    }
  }
  while (!nextWorklist.empty()) {
    worklist.swap(nextWorklist);
    while (!worklist.empty()) {
      auto element = worklist.pop_back_val();
      for (auto* user : element.getUsers()) {
        for (auto const& result : user->getResults()) {
          reachableUses[element].insert(std::begin(reachableUses[result]), std::end(reachableUses[result]));
          reachableUses[element].insert(std::begin(valueOccurrences[result]), std::end(valueOccurrences[result]));
        }
      }
      if (auto* definingOp = element.getDefiningOp()) {
        for (auto const& operand : definingOp->getOperands()) {
          nextWorklist.insert(operand);
        }
      }
    }
  }
  for (auto* node : dependencyGraph.nodes) {
    for (auto const& element : *node) {
      for (auto const& reachableUse : reachableUses[element]) {
        dependencyGraph.dependencyEdges[node].insert(reachableUse);
      }
    }
  }
  return dependencyGraph;
}

// === DependencyGraph === //

size_t DependencyGraph::numNodes() const {
  return nodes.size();
}

size_t DependencyGraph::numEdges() const {
  size_t numEdges = 0;
  for (auto& entry : dependencyEdges) {
    numEdges += entry.second.size();
  }
  return numEdges;
}

SmallVector<Superword*> DependencyGraph::postOrder() const {
  SmallVector<Superword*> order{std::begin(nodes), std::end(nodes)};
  DenseMap<Superword*, unsigned> destinationCounts;
  for (auto* superword : nodes) {
    for (auto* dependency : dependencyEdges.lookup(superword)) {
      ++destinationCounts[dependency];
    }
  }
  llvm::sort(std::begin(order), std::end(order), [&](Superword* lhs, Superword* rhs) {
    if (dependencyEdges.lookup(lhs).contains(rhs)) {
      return true;
    }
    if (dependencyEdges.lookup(rhs).contains(lhs)) {
      return false;
    }
    return destinationCounts[lhs] < destinationCounts[rhs];
  });
  return order;
}