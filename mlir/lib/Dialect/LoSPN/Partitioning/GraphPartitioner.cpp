//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "GraphPartitioner.h"
#include <stack>

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

GraphPartitioner::GraphPartitioner(int maxTaskSize, HeuristicFactory heuristic) :
    maxPartitionSize{maxTaskSize}, factory{std::move(heuristic)} {}

unsigned int GraphPartitioner::getMaximumPartitionSize() const {
  // Allow up to 1% or at least one node in slack.
  unsigned slack = std::max(1u, static_cast<unsigned>(static_cast<double>(maxPartitionSize) * 0.01));
  return maxPartitionSize + slack;
}

Partitioning mlir::spn::low::GraphPartitioner::partitionGraph(
    llvm::ArrayRef<Operation*> nodes,
    llvm::SmallPtrSetImpl<Operation*>& inNodes,
    llvm::ArrayRef<Value> externalInputs) {
  auto partitioning = initialPartitioning(nodes, inNodes, externalInputs);
  refinePartitioning(nodes, externalInputs, &partitioning);
  return partitioning;
}

Partitioning mlir::spn::low::GraphPartitioner::initialPartitioning(
    llvm::ArrayRef<Operation*> nodes,
    llvm::SmallPtrSetImpl<Operation*>& inNodes,
    llvm::ArrayRef<Value> externalInputs) const {
  llvm::SmallPtrSet<Operation*, 32> partitioned;
  std::stack<Operation*> S;
  llvm::SmallVector<Operation*, 0> T;
  llvm::SmallPtrSet<Value, 32> external(externalInputs.begin(), externalInputs.end());
  llvm::SmallVector<Operation*> inputNodes(inNodes.begin(), inNodes.end());
  // Initially populate the stack with all operations that potentially have an in-degree of zero.
  for (auto I = inputNodes.rbegin(); I != inputNodes.rend(); ++I) {
    if (hasInDegreeZero(*I, partitioned, external)) {
      S.push(*I);
    }
  }
  // Iterate all nodes, creating a topological sort order.
  // By using a stack instead of a queue, we effectively create more vertical cuts rather than
  // horizontal cuts with many edges crossing partitions.
  while (T.size() < nodes.size()) {
    assert(!S.empty());
    // Pop the top-most element from the stack.
    auto cur = S.top();
    S.pop();
    T.push_back(cur);
    partitioned.insert(cur);
    // Check if any of the users of this operation have all their operands visited now and
    // push them onto the stack.
    for (auto r : cur->getResults()) {
      for (auto* U : r.getUsers()) {
        if (hasInDegreeZero(U, partitioned, external)) {
          S.push(U);
        }
      }
    }
  }
  // Create partitions from the topological sort order by taking
  // chunks of n nodes from the list and putting them into one partition.
  auto numPartitions = llvm::divideCeil(T.size(), maxPartitionSize);
  Partitioning partitioning;
  unsigned nodeIndex = 0;
  for (unsigned i = 0; i < numPartitions; ++i) {
    partitioning.push_back(std::make_unique<Partition>(i, getMaximumPartitionSize()));
    auto& curPar = partitioning.back();
    auto maxIndex = nodeIndex + maxPartitionSize;
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