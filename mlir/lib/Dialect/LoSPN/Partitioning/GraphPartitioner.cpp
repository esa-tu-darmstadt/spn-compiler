//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "GraphPartitioner.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::spn::low;

void mlir::spn::low::Partition::addNode(Operation* node) {
  nodes.insert(node);
  dirty = true;
  if (!node->hasTrait<OpTrait::ConstantLike>()) {
    ++numNodes;
  }
}

void mlir::spn::low::Partition::removeNode(Operation* node) {
  nodes.erase(node);
  dirty = true;
  if (!node->hasTrait<OpTrait::ConstantLike>()) {
    --numNodes;
  }
}

bool mlir::spn::low::Partition::contains(Operation* node) {
  return nodes.contains(node);
}

llvm::SmallPtrSetImpl<mlir::Operation*>::iterator mlir::spn::low::Partition::begin() {
  return nodes.begin();
}

llvm::SmallPtrSetImpl<mlir::Operation*>::iterator mlir::spn::low::Partition::end() {
  return nodes.end();
}

void mlir::spn::low::Partition::computeExternalConnections() {
  for (auto* n : nodes) {
    auto usesExt = llvm::any_of(n->getOperands(), [this](Value op) {
      return !this->nodes.contains(op.getDefiningOp());
    });
    if (usesExt) {
      extIn.push_back(n);
    }
    auto providesExt = llvm::any_of(n->getUsers(), [this](Operation* u) {
      return !this->nodes.contains(u);
    });
    if (providesExt) {
      exOut.push_back(n);
    }
  }
  dirty = false;
}

llvm::ArrayRef<Operation*> mlir::spn::low::Partition::hasExternalInputs() {
  if (dirty) {
    computeExternalConnections();
  }
  return extIn;
}

llvm::ArrayRef<Operation*> mlir::spn::low::Partition::hasExternalOutputs() {
  if (dirty) {
    computeExternalConnections();
  }
  return exOut;
}

void mlir::spn::low::Partition::dump() const {
  llvm::dbgs() << "Partition " << id << "(" << this << "):\n";
  for (auto* o : nodes) {
    o->dump();
  }
}

GraphPartitioner::GraphPartitioner(unsigned int numberOfPartitions, HeuristicFactory heuristic) :
    numPartitions{numberOfPartitions}, factory{std::move(heuristic)} {}

Partitioning mlir::spn::low::GraphPartitioner::partitionGraph(
    llvm::ArrayRef<Operation*> nodes,
    llvm::ArrayRef<Operation*> inNodes,
    llvm::ArrayRef<Value> externalInputs) {
  auto partitioning = initialPartitioning(nodes, inNodes, externalInputs);
  refinePartitioning(nodes, externalInputs, &partitioning);
  return partitioning;
  // TODO: Special handling of constant operations. They can simply be duplicated if necessary and should
  // never imply an edge crossing partitions.
}

Partitioning mlir::spn::low::GraphPartitioner::initialPartitioning(
    llvm::ArrayRef<Operation*> nodes,
    llvm::ArrayRef<Operation*> inNodes,
    llvm::ArrayRef<Value> externalInputs) const {
  llvm::SmallPtrSet<Operation*, 32> partitioned;
  llvm::SmallVector<Operation*> S;
  llvm::SmallVector<Operation*, 0> T;
  llvm::SmallPtrSet<Value, 32> external(externalInputs.begin(), externalInputs.end());
  for (auto I = inNodes.rbegin(); I != inNodes.rend(); ++I) {
    if (hasInDegreeZero(*I, partitioned, external)) {
      S.push_back(*I);
    }
  }
  while (T.size() < nodes.size()) {
    assert(!S.empty());
    auto cur = S.pop_back_val();
    T.push_back(cur);
    partitioned.insert(cur);
    for (auto r : cur->getResults()) {
      for (auto* U : r.getUsers()) {
        if (hasInDegreeZero(U, partitioned, external)) {
          S.push_back(U);
        }
      }
    }
  }
  // TODO Make this configurable
  auto nodesPerPartition = llvm::divideNearest(T.size(), numPartitions);
  llvm::dbgs() << "Nodes per Partition: " << nodesPerPartition << "\n";
  Partitioning partitioning;
  unsigned nodeIndex = 0;
  for (unsigned i = 0; i < numPartitions; ++i) {
    partitioning.push_back(std::make_unique<Partition>(i, nodesPerPartition));
    auto& curPar = partitioning.back();
    auto maxIndex = nodeIndex + nodesPerPartition;
    for (; (nodeIndex < maxIndex) && (nodeIndex < T.size()); ++nodeIndex) {
      curPar->addNode(T[nodeIndex]);
    }
  }
  return partitioning;
}

bool mlir::spn::low::GraphPartitioner::hasInDegreeZero(Operation* node,
                                                       llvm::SmallPtrSetImpl<Operation*>& partitioned,
                                                       llvm::SmallPtrSetImpl<Value>& externalInputs) const {
  return llvm::all_of(node->getOperands(), [&](Value operand) {
    return externalInputs.contains(operand)
        || (operand.getDefiningOp() && partitioned.contains(operand.getDefiningOp()));
  });
}

void GraphPartitioner::refinePartitioning(llvm::ArrayRef<Operation*> allNodes,
                                          llvm::ArrayRef<Value> externalInputs,
                                          Partitioning* allPartitions) {
  if (!factory) {
    return;
  }
  auto heuristic = factory(allNodes, externalInputs, allPartitions);
  heuristic->refinePartitioning();
}