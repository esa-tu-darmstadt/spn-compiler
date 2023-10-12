#include "TopoSortPartitioner.h"
#include <stack>

using namespace mlir::spn::low::partitioning;

Partitioning TopoSortPartitioner::partitionGraph() {

  auto partitioning = initialPartitioning();

  viewGraph(partitioning);

  refinePartitioning(&partitioning);
  return partitioning;
}

Partitioning TopoSortPartitioner::initialPartitioning() const {
  std::set<Node> partitioned;
  std::stack<Operation *> S;
  llvm::SmallVector<Operation *, 0> T;

  // Initially populate the stack with all operations that potentially have an in-degree of zero.
  for (auto leaf : llvm::reverse(leaf_nodes())) {
    S.push(leaf);
  }

  // Iterate all nodes, creating a topological sort order.
  // By using a stack instead of a queue, we effectively create more vertical cuts rather than
  // horizontal cuts with many edges crossing partitions.
  while (T.size() < nodes().size()) {
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
        if (hasInDegreeZero(U, partitioned)) {
          S.push(U);
        }
      }
    }
  }
  // Create partitions from the topological sort order by taking
  // chunks of n nodes from the list and putting them into one partition.
  auto numPartitions = llvm::divideCeil(T.size(), maxPartitionSize_);
  Partitioning partitioning;
  unsigned nodeIndex = 0;
  for (unsigned i = 0; i < numPartitions; ++i) {
    partitioning.push_back(std::make_unique<Partition>(i, getMaximumPartitionSize()));
    auto &curPar = partitioning.back();
    auto maxIndex = nodeIndex + maxPartitionSize_;
    for (; (nodeIndex < maxIndex) && (nodeIndex < T.size()); ++nodeIndex) {
      curPar->addNode(T[nodeIndex]);
    }
  }
  return partitioning;
}

void TopoSortPartitioner::refinePartitioning(Partitioning *allPartitions) {
  if (!factory) {
    return;
  }
  // auto heuristic = factory(allNodes, externalInputs, allPartitions);
  // heuristic->refinePartitioning();
}

bool TopoSortPartitioner::hasInDegreeZero(Node node, std::set<Node> &partitioned) const {
  // Return true if the node is a leaf or all of its operands are already partitioned.
  return node.isLeaf() || llvm::all_of(node->getOperands(), [&](Value operand) {
           return operand.getDefiningOp() && partitioned.find(operand.getDefiningOp()) != partitioned.end();
         });
}