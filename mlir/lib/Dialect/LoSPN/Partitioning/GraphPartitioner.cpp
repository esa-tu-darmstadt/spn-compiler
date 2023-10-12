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
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::spn::low;
using namespace mlir::spn::low::partitioning;

Node Edge::from() const { return Node(from_); }
Node Edge::to() const { return Node(to_); }

float Edge::getCost() const { return 10.0; }

float Node::getCost() const {
  if (op_->hasTrait<OpTrait::ConstantLike>()) {
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

GraphPartitioner::GraphPartitioner(llvm::ArrayRef<Node> nodes, int maxTaskSize)
    : maxPartitionSize_{maxTaskSize}, nodes_(nodes) {}

unsigned int GraphPartitioner::getMaximumPartitionSize() const {
  // Allow up to 1% or at least one node in slack.
  unsigned slack = std::max(1u, static_cast<unsigned>(static_cast<double>(maxPartitionSize_) * 0.01));
  return maxPartitionSize_ + slack;
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
      O << "    \"" << node << "\" [label=\"" << node->getName() << " (" << node.getOperation() << ")\"];\n";
    }

    // Write the edges between the nodes of this partition.
    for (auto &node : *partition) {
      for (auto succ : node.edges_out()) {
        if (partition->contains(succ.to()))
          O << "    \"" << *node << "\" -> \"" << *succ.to() << "\";\n";
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