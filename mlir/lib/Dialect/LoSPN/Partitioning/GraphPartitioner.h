//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>
#include <set>

namespace mlir {
namespace spn {
namespace low {

namespace partitioning {

/// A node in the SPN graph.
class Node {
  Operation *op;

public:
  Node(Operation *op) : op(op) {}

  /// Range of all users of this operation wrapped in Node objects.
  auto successors() const {
    return llvm::map_range(op->getUsers(), [](Operation *op) { return Node(op); });
  }

  /// Range of all operands of this operation that are results of an operation wrapped in Node objects.
  auto predecessors() const {
    return llvm::map_range(llvm::make_filter_range(op->getOperands(), [](Value op) { return op.getDefiningOp(); }),
                           [](Value op) { return Node(op.getDefiningOp()); });
  }

  /// Range of all external inputs, i.e., all operands that are block arguments.
  auto external_inputs() const {
    return llvm::map_range(
        llvm::make_filter_range(op->getOperands(), [](Value op) { return op.isa<mlir::BlockArgument>(); }),
        [](Value op) { return op.cast<mlir::BlockArgument>(); });
  }

  Operation *getOperation() const { return op; }
  Operation *operator->() const { return op; }
  Operation *operator*() const { return op; }

  // Allows implicit conversion to Operation *.
  // This allows for an easy use of this class in sets and maps and automatically brings comparison operators with it.
  operator Operation *() const { return op; }

  float getWeight() const;
};

/// A partition of the SPN graph.
class Partition {
  std::set<Node> nodes_;

public:
  Partition(unsigned ID, unsigned maximumSize) : id{ID}, numNodes{0}, sizeBoundary{maximumSize} {};

  /// Range of all nodes in this partition.
  auto nodes() const { return llvm::make_range(nodes_.begin(), nodes_.end()); }
  auto begin() const { return nodes_.begin(); }
  auto end() const { return nodes_.end(); }

  /// Range of nodes with operands connected to nodes in other partitions.
  auto incoming_nodes() const {
    return llvm::make_filter_range(nodes(), [this](Node n) {
      return llvm::any_of(n.predecessors(), [this](Node predecessor) { return !this->contains(predecessor); });
    });
  }

  /// Range of nodes with users connected to nodes in other partitions.
  auto outgoing_nodes() const {
    return llvm::make_filter_range(nodes(), [this](Node n) {
      return llvm::any_of(n.successors(), [this](Node successor) { return !this->contains(successor); });
    });
  }

  /// Range of nodes with operands connected to block arguments, i.e., input values from the user.
  auto external_input_nodes() const {
    return llvm::make_filter_range(nodes(), [](Node node) {
      return llvm::any_of(node->getOperands(), [](Value op) { return op.isa<mlir::BlockArgument>(); });
    });
  }

  void addNode(Node node) { nodes_.insert(node); }
  void removeNode(Node node) { nodes_.erase(node); }
  bool contains(Node node) const { return llvm::find(nodes(), node) != nodes().end(); }

  unsigned ID() const { return id; }

  unsigned size() const { return numNodes; }

  bool canAccept() const { return numNodes < sizeBoundary; }

  void dump() const;

private:
  unsigned id;

  unsigned numNodes;

  unsigned sizeBoundary;
};

using PartitionRef = std::unique_ptr<Partition>;
using Partitioning = std::vector<PartitionRef>;

class Heuristic;
using HeuristicFactory =
    std::function<std::unique_ptr<Heuristic>(llvm::ArrayRef<Node>, llvm::ArrayRef<Value>, Partitioning *)>;

class GraphPartitioner {

public:
  explicit GraphPartitioner(int maxTaskSize, HeuristicFactory heuristic = nullptr);

  void viewGraph(const Partitioning &partitions) const;
  void printGraph(raw_ostream &O, const Partitioning &partitions) const;

  Partitioning partitionGraph(llvm::ArrayRef<Operation *> nodes, llvm::SmallPtrSetImpl<Operation *> &inNodes,
                              llvm::ArrayRef<Value> externalInputs);

  unsigned getMaximumPartitionSize() const;

private:
  Partitioning initialPartitioning(llvm::ArrayRef<Operation *> nodes, llvm::SmallPtrSetImpl<Operation *> &inNodes,
                                   llvm::ArrayRef<Value> externalInputs) const;

  void refinePartitioning(llvm::ArrayRef<Operation *> allNodes, llvm::ArrayRef<Value> externalInputs,
                          Partitioning *allPartitions);

  bool hasInDegreeZero(Operation *node, llvm::SmallPtrSetImpl<Operation *> &partitioned,
                       llvm::SmallPtrSetImpl<Value> &externalInputs) const;

  int maxPartitionSize;

  HeuristicFactory factory;
};

} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir

// namespace std {
// /// Hash function for Node objects that hashes the underlying operation pointer.
// template <> struct hash<mlir::spn::low::partitioning::Node> {
//   std::size_t operator()(const mlir::spn::low::partitioning::Node &node) const {
//     return hash<mlir::Operation *>()(node.getOperation());
//   }
// };
// } // namespace std

#endif // SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
