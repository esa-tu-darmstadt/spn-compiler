//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/iterator_range.h"
#include <functional>
#include <set>

namespace mlir {
namespace spn {
namespace low {

namespace partitioning {
class Node;

class Edge {
  Operation *from_;
  Operation *to_;
  Value value_;

public:
  Edge(Operation *from, Operation *to, Value value) : from_(from), to_(to), value_(value) {}

  Node from() const;
  Node to() const;
  Value getValue() const { return value_; }

  // Allows implicit conversion to Value.
  operator Value() { return value_; }
  Value operator->() const { return value_; }
  Value operator*() const { return value_; }

  // Returns the cost to transmit this edge to another partition.
  float getCost() const;
};

/// A node in the SPN graph.
class Node {
  Operation *op_;

public:
  Node() : op_(nullptr) {}
  Node(Operation *op) : op_(op) {}

  /// Range of all outgoing edges
  auto edges_out() const {
    assert(op_->getNumResults() == 1 && "Operation must have exactly one result.");
    return llvm::map_range(op_->getResult(0).getUsers(),
                           [this](Operation *user) { return Edge(op_, user, op_->getResult(0)); });
  }

  /// Range of all incoming edges
  auto edges_in() const {
    return llvm::map_range(
        llvm::make_filter_range(op_->getOperands(), [](Value operand) { return operand.getDefiningOp(); }),
        [this](Value operand) { return Edge(operand.getDefiningOp(), op_, operand); });
  }

  /// Range of all external input edges, i.e., block arguments.
  auto external_edges_in() const {
    return llvm::map_range(
        llvm::make_filter_range(op_->getOperands(), [](Value operand) { return operand.isa<mlir::BlockArgument>(); }),
        [this](Value operand) { return Edge(nullptr, op_, operand); });
  }

  /// Returns whether this node is a leaf node.
  bool isLeaf() const {
    return op_->hasTrait<OpTrait::ConstantLike>() ||
           llvm::any_of(op_->getOperands(), [](Value op) { return op.isa<mlir::BlockArgument>(); });
  }

  bool isRoot() const { return op_-> }

  Operation *getOperation() const { return op_; }
  Operation *operator->() const { return op_; }
  Operation *operator*() const { return op_; }

  // Allows implicit conversion to Operation *.
  // This allows for an easy use of this class in sets and maps and automatically brings comparison operators with it.
  operator Operation *() const { return op_; }

  // Returns the cost to compute this node.
  float getCost() const;
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

  /// Range of nodes with edges connected to nodes in other partitions.
  auto incoming_nodes() const {
    return llvm::make_filter_range(nodes(), [this](Node n) {
      return llvm::any_of(n.edges_in(), [this](Edge edge) { return !this->contains(edge.from()); });
    });
  }

  /// Range of nodes with edges connected to nodes in other partitions.
  auto outgoing_nodes() const {
    return llvm::make_filter_range(nodes(), [this](Node n) {
      return llvm::any_of(n.edges_out(), [this](Edge edge) { return !this->contains(edge.to()); });
    });
  }

  /// Range of entry nodes with an indegree of zero. These are either constant nodes or nodes with external inputs.
  auto leaf_nodes() const {
    return llvm::make_filter_range(nodes(), [](Node node) { return node.isLeaf(); });
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

class GraphPartitioner {

public:
  explicit GraphPartitioner(llvm::ArrayRef<Node> nodes, int maxTaskSize);
  virtual ~GraphPartitioner() = default;

  void viewGraph(const Partitioning &partitions) const;
  void printGraph(raw_ostream &O, const Partitioning &partitions) const;

  virtual Partitioning partitionGraph() = 0;

  unsigned getMaximumPartitionSize() const;

  /// Range of all nodes in the graph.
  auto nodes() const { return nodes_; }

  /// Range of all leaf nodes in the graph.
  auto leaf_nodes() const {
    return llvm::make_filter_range(nodes(), [](Node node) { return node.isLeaf(); });
  }

protected:
  int maxPartitionSize_;
  llvm::ArrayRef<Node> nodes_;
};

} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir

namespace std {
/// Hash function for Node objects that hashes the underlying operation pointer.
/// This allows to use Node objects in sets and maps.
template <> struct hash<mlir::spn::low::partitioning::Node> {
  std::size_t operator()(const mlir::spn::low::partitioning::Node &node) const {
    return hash<mlir::Operation *>()(node.getOperation());
  }
};
} // namespace std

#endif // SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
