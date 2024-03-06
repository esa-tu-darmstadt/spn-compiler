//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace spn {
namespace low {
namespace slp {

/// A superword models an SLP vector, which basically consists of a fixed amount
/// of operations (depending on the hardware's supported vector width) that can
/// be computed isomorphically. Every superword tracks its operand superwords,
/// i.e. those that it would use for computation of its own elements. The term
/// 'superword' was chosen because a 'vector' is quite overloaded in C++.
class Superword {

  friend class SLPNode;

public:
  explicit Superword(ArrayRef<Value> values);
  explicit Superword(ArrayRef<Operation *> operations);

  /// Retrieve the superword's value in the requested lane.
  Value getElement(size_t lane) const;
  /// Replace the superword's value in the requested lane with the provided
  /// value.
  void setElement(size_t lane, Value value);
  /// Retrieve the superword's value in the requested lane.
  Value operator[](size_t lane) const;

  /// Check if the superword contains the requested value.
  bool contains(Value value) const;
  /// Check if the superword is a leaf superword, i.e. if it does not have any
  /// superword operands.
  bool isLeaf() const;
  /// Check if the superword consists of constant operations only.
  bool constant() const;

  /// Return the number of lanes of the superword (its vector width).
  size_t numLanes() const;
  /// Return an iterator pointing to the first element of the superword.
  SmallVectorImpl<Value>::const_iterator begin() const;
  /// Return an iterator pointing after the last element of the superword.
  SmallVectorImpl<Value>::const_iterator end() const;

  /// Return the number of superword operands.
  size_t numOperands() const;
  /// Append an operand superword to the internal list of operand superwords.
  void addOperand(std::shared_ptr<Superword> operandWord);
  /// Retrieve the operand superword at the index's position (the index-th
  /// operand).
  Superword *getOperand(size_t index) const;
  /// Retrieve all operands of the superword.
  SmallVector<Superword *, 2> getOperands() const;

  /// Check if the superword has had its semantics altered in the requested
  /// lane. This can happen if the computation chain of its scalar element in
  /// that lane does not correspond to the computation chain of the elements in
  /// the superword's operand chain anymore (e.g. through reordering of elements
  /// in the superword's operands).
  bool hasAlteredSemanticsInLane(size_t lane) const;
  /// Mark the element in the specific lane as semantically altered.
  void markSemanticsAlteredInLane(size_t lane);

  /// Returns a MLIR vector type corresponding to the superword.
  VectorType getVectorType() const;
  /// Return the MLIR type of the superword's elements.
  Type getElementType() const;
  /// Return the location of the superword in the program. This always
  /// corresponds to the location of the first element.
  Location getLoc() const;

private:
  SmallVector<Value, 4> values;
  SmallVector<std::shared_ptr<Superword>, 2> operandWords;
  /// Stores a bit for each lane. If the bit is set to true, the semantics of
  /// that lane have been altered and the value that is present there is not
  /// actually being computed anymore.
  llvm::BitVector semanticsAltered;
};

/// The dependency graph models flow dependencies of an SLP graph. Its nodes are
/// superwords. Edges exist between superwords if at least one element of the
/// source superword 'flows into' at least one element in the destination
/// superword (i.e. it appears in the computation chain of the destination
/// element).
struct DependencyGraph {
  /// Return the number of nodes in the dependency graph.
  size_t numNodes() const;
  /// Return the number of edges in the dependency graph.
  size_t numEdges() const;
  /// Return the dependency graph's superwords in post order. Here, a superword
  /// s1 always appears before an other superword s2 if there is no dependency
  /// edge from s2 to s1. If there is no edge between them, the superwords are
  /// sorted by how often they are used as destination of an edge.
  SmallVector<Superword *> postOrder() const;
  /// The superwords contained in the dependency graph.
  SmallPtrSet<Superword *, 32> nodes;
  /// A mapping of superwords to their outgoing edges.
  DenseMap<Superword *, SmallPtrSet<Superword *, 1>> dependencyEdges;
};

/// This class models the nodes in an SLP graph. Each SLP node consists of one
/// or more superwords and can have zero or more operand nodes. It can be
/// understood as a kind of superstructure around the graph modeled by the
/// superwords themselves.
class SLPNode {

public:
  explicit SLPNode(std::shared_ptr<Superword> superword);

  /// Add a superword to the node. This turns the node into a 'multinode'.
  void addSuperword(std::shared_ptr<Superword> superword);
  /// Retrieve the node's superword at the requested index.
  std::shared_ptr<Superword> getSuperword(size_t index) const;

  /// Retrieve the element of the superword at the requested index and lane.
  Value getValue(size_t lane, size_t index) const;
  /// Replace the element of the superword at the requested index and lane with
  /// the provided value.
  void setValue(size_t lane, size_t index, Value newValue);

  /// Check if the node contains the requested value.
  bool contains(Value value) const;

  /// Check if the provided superword is the root superword of the node (the
  /// very first one).
  bool isSuperwordRoot(Superword const &superword) const;

  /// Return the number of lanes in the node (i.e. its vector width).
  size_t numLanes() const;
  /// Return the number of superwords in the node.
  size_t numSuperwords() const;
  /// Return the number of operand nodes.
  size_t numOperands() const;

  /// Add an operand node to the node.
  void addOperand(std::shared_ptr<SLPNode> operandNode);
  /// Return the operand at the requested index.
  SLPNode *getOperand(size_t index) const;
  /// Return all operands of the node.
  ArrayRef<std::shared_ptr<SLPNode>> getOperands() const;

private:
  SmallVector<std::shared_ptr<Superword>> superwords;
  SmallVector<std::shared_ptr<SLPNode>> operandNodes;
};

/// The SLP graph models a graph whose nodes are SLP nodes and whose edges are
/// computation flows between the nodes' superwords.
class SLPGraph {
  friend class SLPGraphBuilder;

public:
  /// Construct a new SLP graph based on the provided seed.
  SLPGraph(ArrayRef<Value> seed, unsigned maxNodeSize, unsigned maxLookAhead,
           bool allowDuplicateElements, bool allowTopologicalMixing,
           bool useXorChains);
  /// Return the very last superword of the graph's computation chain (i.e. the
  /// one that contains the seed operations).
  std::shared_ptr<Superword> getRootSuperword() const;
  /// Return the very last SLP node of the graph's computation chain.
  std::shared_ptr<SLPNode> getRootNode() const;
  /// Construct a dependency graph based on the graph.
  DependencyGraph dependencyGraph() const;

private:
  std::shared_ptr<Superword> superwordRoot;
  std::shared_ptr<SLPNode> nodeRoot;
};

namespace graph {

/// Walks through a graph rooted at node 'root' in post order and applies
/// function 'f' to every visited node.
template <typename Node, typename Function>
void walk(Node *root, Function f) {
  // Use a stack-based approach instead of recursion. Whenever a (node, true)
  // pair is popped from the stack, the function is called on the node since
  // then its operands have all been handled already. TL;DR: false = visit node
  // operands first, true = finished
  SmallVector<std::pair<Node *, bool>> worklist;
  // Prevents looping endlessly in case a graph contains loops.
  llvm::SmallSet<Node *, 32> finishedNodes;
  worklist.emplace_back(root, false);
  while (!worklist.empty()) {
    if (finishedNodes.contains(worklist.back().first)) {
      worklist.pop_back();
      continue;
    }
    auto *node = worklist.back().first;
    bool operandsDone = worklist.back().second;
    worklist.pop_back();
    if (operandsDone) {
      finishedNodes.insert(node);
      f(node);
    } else {
      worklist.emplace_back(node, true);
      for (size_t i = node->numOperands(); i-- > 0;) {
        worklist.emplace_back(node->getOperand(i), false);
      }
    }
  }
}

/// Construct a post-order version of the graph rooted at the provided node.
template <typename Node>
SmallVector<Node *> postOrder(Node *root) {
  SmallVector<Node *> order;
  walk(root, [&](Node *node) { order.template emplace_back(node); });
  return order;
}

} // namespace graph
} // namespace slp
} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
