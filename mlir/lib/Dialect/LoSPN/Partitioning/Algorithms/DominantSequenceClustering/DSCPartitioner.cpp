#include "DSCPartitioner.h"

using namespace mlir::spn::low::partitioning;

float DSCPartitioner::findLongestPath(Node node, std::vector<Node> &path) {
  if (visited_[node] || node.isRoot()) {
    return 0.0f;
  }

  visited_[node] = true;
  float max_cost = node.getCost();
  Node next_in_sequence; // To track the node leading to the longest path

  for (auto edge : node.edges_out()) {
    float current_cost = edge.getCost() + findLongestPath(edge.to(), path);
    if (current_cost > max_cost) {
      max_cost = current_cost;
      next_in_sequence = edge.to();
    }
  }

  path.push_back(next_in_sequence); // Add the node to the dominant sequence
  return max_cost;
}

Partitioning DSCPartitioner::partitionGraph() {
  Partitioning result;

  // Step 1: Identify the dominant sequence
  std::vector<Node> dominantSequence;
  for (auto node : leaf_nodes()) {
    visited_.clear();
    findLongestPath(node, dominantSequence);
  }

  // Step 2: Cluster formation
  int nextPartitionID = 0;
  PartitionRef currentPartition = std::make_unique<Partition>(nextPartitionID++, maxPartitionSize_);
  float currentCost = 0.0f;

  for (auto node : dominantSequence) {
    if ((currentCost + node.getCost()) <= getMaximumPartitionSize()) {
      currentPartition->addNode(node);
      currentCost += node.getCost();
    } else {
      result.push_back(std::move(currentPartition));
      currentPartition = std::make_unique<Partition>(nextPartitionID++, maxPartitionSize_);
      currentPartition->addNode(node);
      currentCost = node.getCost();
    }
  }

  // Handle any remaining nodes in the current partition
  if (currentPartition->size() > 0) {
    result.push_back(std::move(currentPartition));
  }

  return result;
}