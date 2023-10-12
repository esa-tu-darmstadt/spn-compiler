//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "GraphPartitioner.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/GraphWriter.h"
#include <stack>

using namespace llvm;
using namespace mlir;
using namespace mlir::spn::low;
using namespace mlir::spn::low::partitioning;

float Node::getWeight() const {
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    return 0.0;
  }
  return 1.0;
}

void partitioning::Partition::dump() const {
  llvm::dbgs() << "Partition " << id << "(" << this << "):\n";
  for (auto &o : nodes_) {
    o->dump();
  }
}

GraphPartitioner::GraphPartitioner(int maxTaskSize, HeuristicFactory heuristic)
    : maxPartitionSize{maxTaskSize}, factory{std::move(heuristic)} {}

unsigned int GraphPartitioner::getMaximumPartitionSize() const {
  // Allow up to 1% or at least one node in slack.
  unsigned slack = std::max(1u, static_cast<unsigned>(static_cast<double>(maxPartitionSize) * 0.01));
  return maxPartitionSize + slack;
}

Partitioning GraphPartitioner::partitionGraph(llvm::ArrayRef<Operation *> nodes,
                                              llvm::SmallPtrSetImpl<Operation *> &inNodes,
                                              llvm::ArrayRef<Value> externalInputs) {

  auto partitioning = initialPartitioning(nodes, inNodes, externalInputs);

  viewGraph(partitioning);

  refinePartitioning(nodes, externalInputs, &partitioning);
  return partitioning;
}

Partitioning GraphPartitioner::initialPartitioning(llvm::ArrayRef<Operation *> nodes,
                                                   llvm::SmallPtrSetImpl<Operation *> &inNodes,
                                                   llvm::ArrayRef<Value> externalInputs) const {
  llvm::SmallPtrSet<Operation *, 32> partitioned;
  std::stack<Operation *> S;
  llvm::SmallVector<Operation *, 0> T;
  llvm::SmallPtrSet<Value, 32> external(externalInputs.begin(), externalInputs.end());
  llvm::SmallVector<Operation *> inputNodes(inNodes.begin(), inNodes.end());
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
      for (auto *U : r.getUsers()) {
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
    auto &curPar = partitioning.back();
    auto maxIndex = nodeIndex + maxPartitionSize;
    for (; (nodeIndex < maxIndex) && (nodeIndex < T.size()); ++nodeIndex) {
      curPar->addNode(T[nodeIndex]);
    }
  }
  return partitioning;
}

bool GraphPartitioner::hasInDegreeZero(Operation *node, llvm::SmallPtrSetImpl<Operation *> &partitioned,
                                       llvm::SmallPtrSetImpl<Value> &externalInputs) const {
  return llvm::all_of(node->getOperands(), [&](Value operand) {
    return externalInputs.contains(operand) ||
           (operand.getDefiningOp() && partitioned.contains(operand.getDefiningOp()));
  });
}

void GraphPartitioner::refinePartitioning(llvm::ArrayRef<Operation *> allNodes, llvm::ArrayRef<Value> externalInputs,
                                          Partitioning *allPartitions) {
  if (!factory) {
    return;
  }
  // auto heuristic = factory(allNodes, externalInputs, allPartitions);
  // heuristic->refinePartitioning();
}

void GraphPartitioner::viewGraph(const Partitioning &partitions) const {
  int FD;

  // Create a temporary file to hold the graph.
  auto fileName = createGraphFilename("partitioning", FD);
  if (fileName.empty()) {
    return;
  }

  // Open the file for writing.
  raw_fd_ostream O(FD, /*shouldClose=*/true);
  if (FD == -1) {
    errs() << "error opening file '" << fileName << "' for writing!\n";
    return;
  }

  // Write the graph to the file.
  printGraph(O, partitions);
  O.close();

  // Display the graph.
  DisplayGraph(fileName, false, GraphProgram::DOT);
}

void GraphPartitioner::printGraph(raw_ostream &O, const Partitioning &partitions) const {
  // Write the header
  O << "digraph \"Partitioning\" {\n";

  // Write each partition into a subgraph.
  for (auto &partition : partitions) {
    // Write the header of the subgraph.
    O << "  subgraph cluster_" << partition->ID() << " {\n";
    O << "    label = \"Partition " << partition->ID() << "\";\n";

    // Write each operation as a node. Use the address of the operation as its identifier and its name as the label.
    for (auto &node : *partition) {
      O << "    \"" << *node << "\" [label=\"" << node->getName() << "\"];\n";
    }

    // Write the edges between the nodes of this partition.
    for (auto &node : *partition) {
      for (auto succ : node.successors()) {
        if (partition->contains(succ))
          O << "    \"" << *node << "\" -> \"" << *succ << "\";\n";
      }
    }

    // Write the footer of the subgraph.
    O << "  }\n";
  }

  // Write the block arguments as nodes.
  for (auto &partition : partitions) {
    for (auto &node : partition->incoming_nodes()) {
      for (const auto &operand : node->getOperands()) {
        if (auto ba = operand.dyn_cast_or_null<mlir::BlockArgument>())
          O << "  \"" << operand.getAsOpaquePointer() << "\" [label=\"Block arg #" << ba.getArgNumber() << "\"];\n";
      }
    }
  }

  // Write the edges between block arguments and nodes.
  for (auto &partition : partitions) {
    for (auto &node : partition->incoming_nodes()) {
      for (const auto &operand : node->getOperands()) {
        if (operand.getDefiningOp() != nullptr)
          continue;
        O << "  \"" << operand.getAsOpaquePointer() << "\" -> \"" << *node << "\";\n";
      }
    }
  }

  // Write the edges between nodes of different partitions.
  for (auto &partition : partitions) {
    for (auto &node : partition->outgoing_nodes()) {
      for (auto *user : node->getUsers()) {
        if (!partition->contains(user))
          O << "  \"" << *node << "\" -> \"" << user << "\";\n";
      }
    }
  }

  // Write the footer
  O << "}\n";
}