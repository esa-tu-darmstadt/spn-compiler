//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "DominantSequenceClustering.h"
#include "../../SPNGraph.h"
#include "../../Schedule.h"

#include <boost/graph/subgraph.hpp>
#include <queue>
#include <unordered_map>
#include <vector>

using namespace mlir::spn::low::partitioning;

typedef int level_t;
typedef int cluster_t;

template<typename GraphT>
level_t weight(GraphT &graph, typename GraphT::vertex_descriptor v) { return boost::get(vertex_weight(), graph, v); }
template<typename GraphT>
level_t weight(GraphT &graph, typename GraphT::edge_descriptor e) { return boost::get(edge_weight(), graph, e); }

/// Calculates the longest path from a given node to an exit node. Assumes that every node is in its own cluster.
/// @param graph The graph to calculate the longest path in.
/// @param v The node to calculate the longest path from.
/// @param blevels A vector that maps from a vertex to its blevel. Used to cache the blevel.
template<typename GraphT>
level_t calc_blevel(GraphT &graph, typename GraphT::vertex_descriptor v, std::vector<level_t> &blevels) {
  if (blevels[v] != -1) {
    return blevels[v];
  }

  level_t maxSuccessorBLevel = 0;
  for (auto outedge : boost::make_iterator_range(out_edges(v, graph))) {
    auto successor = target(outedge, graph);
    level_t successorBLevel = calc_blevel(graph, successor, blevels) + weight(graph, outedge);
    if (successorBLevel > maxSuccessorBLevel) {
      maxSuccessorBLevel = successorBLevel;
    }
  }

  blevels[v] = maxSuccessorBLevel + weight(graph, v);
  return blevels[v];
}

/// Calculates the longest path from a given node to an exit node. Assumes that tlevels of predecessors are already
/// calculated.
/// @param graph The graph to calculate the longest path in.
/// @param v The node to calculate the longest path from.
/// @param tlevels A vector that maps from a vertex to its tlevel.
/// @param cluster A vector that maps from a vertex to its cluster index.
template<class GraphT>
void update_tlevel(GraphT &graph, typename GraphT::vertex_descriptor v, std::vector<level_t> &tlevels,
                   std::vector<cluster_t> &cluster) {
  auto ownCluster = cluster[v];
  tlevels[v] = 0;
  for (auto inedge : boost::make_iterator_range(in_edges(v, graph))) {
    auto predecessor = source(inedge, graph);
    auto predecessorCluster = cluster[predecessor];

    level_t tlevelOverPredecessor = tlevels[predecessor] + weight(graph, predecessor);

    if (predecessorCluster != ownCluster) {
      tlevelOverPredecessor += weight(graph, inedge);
    }

    if (tlevelOverPredecessor > tlevels[v]) {
      tlevels[v] = tlevelOverPredecessor;
    }
  }
}

/// Initial implementation of the Dominant Sequence Clustering algorithm.
/// Described in Figure 3 of "DSC: Scheduling Parallel Tasks on an Unbounded Number of Processors"
template <typename GraphT>
void DSC_I(GraphT &graph, std::vector<cluster_t> &cluster, Schedule<GraphT> &schedule) {
  // Calculate blevel
  std::vector<level_t> blevel(num_vertices(graph), -1);
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    blevel[v] = calc_blevel(graph, v, blevel);
    llvm::outs() << "Blevel of " << v << " is " << blevel[v] << "\n";
  }

  // Initialize with tlevel=0 for every (entry) node
  std::vector<level_t> tlevel(num_vertices(graph), -1);
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    if (boost::in_degree(v, graph) == 0) {
      tlevel[v] = 0;
    }
  }

  // Maps from a vertex to its cluster index
  cluster.reserve(num_vertices(graph));

  // Initialize every node into its own cluster
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    cluster[v] = v;
  }

  // List of examined nodes
  std::set<typename GraphT::vertex_descriptor> EG;

  // List of unexamined nodes
  std::set<typename GraphT::vertex_descriptor> UEG;

  // Mark all nodes as unexamined
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    UEG.insert(v);
  }

  // List of free (unexamined) nodes. The compare function compares the priority of the nodes.
  auto priority_comp = [&tlevel, &blevel](SPNGraph::vertex_descriptor a, SPNGraph::vertex_descriptor b) {
    return (tlevel[a] + blevel[a]) < (tlevel[b] + blevel[b]);
  };
  std::priority_queue<SPNGraph::vertex_descriptor, std::vector<SPNGraph::vertex_descriptor>, decltype(priority_comp)>
      FL(priority_comp);

  // Add all free nodes to FL. A node is free if all of its predecessors are examined.
  // At the beginning, all entry nodes are free.
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    if (boost::in_degree(v, graph) == 0) {
      FL.push(v);
    }
  }

  // While there are unexamined nodes
  while (!UEG.empty()) {
    // Find a free node with highest priority from UEG
    // All free nodes are in FL, so we can just take the last element
    SPNGraph::vertex_descriptor nf = FL.top();
    FL.pop();

    llvm::outs() << "Examining " << nf << ". Current tlevel is " << tlevel[nf] << "\n";

    // Merge nf with the cluster of one of its predecessors such that tlevel(nf) decreases in a maximal way.
    // If all zeroing increase tlevel(nf), nf remains in its own cluster.

    // Captures the best predecessor to merge with and the new tlevel of nf
    SPNGraph::vertex_descriptor predecessorToMergeWith = SPNGraph::null_vertex();
    level_t bestTLevel = tlevel[nf];

    for (auto inedge : boost::make_iterator_range(in_edges(nf, graph))) {
      auto predecessor = source(inedge, graph);

      // Calculate the new tlevel of nf if we merge it with predecessor
      // The tlevel would be the tlevel of the predecessor plus the weight of the succesor node
      // The weight of the edge does not matter because we calculate for the case that we merge nf's cluster with
      // predecessor's cluster
      level_t newTLevel = tlevel[predecessor] + weight(graph, predecessor);
      llvm::outs() << "New tlevel of " << nf << " if merged with " << predecessor << " is " << newTLevel << "\n";
      if (newTLevel < bestTLevel) {
        bestTLevel = newTLevel;
        predecessorToMergeWith = predecessor;
      }
    }

    // Merge nf with the cluster of predecessorToMergeWith if we found a predecessor to merge with
    if (predecessorToMergeWith != SPNGraph::null_vertex()) {
      cluster[nf] = cluster[predecessorToMergeWith];
      llvm::outs() << "Merging " << nf << " into cluster " << cluster[nf] << "\n";
    } else {
      llvm::outs() << "Node " << nf << " remains in its own cluster\n";
    }

    // Add nf to the schedule
    schedule[cluster[nf]].push_back(nf);

    // Update the priorities of nf's successors and add them to FL if they just became free
    for (auto outedge : boost::make_iterator_range(out_edges(nf, graph))) {
      auto successor = target(outedge, graph);

      // Update tlevel of the successor
      level_t oldTLevel = tlevel[successor];
      update_tlevel(graph, successor, tlevel, cluster);
      llvm::outs() << "New tlevel of " << successor << " is " << tlevel[successor] << "\n";

      // If all predecessors of successor are examined, the node is considered free and we add it to FL
      bool allPredecessorsExamined = true;
      for (auto inedge : boost::make_iterator_range(in_edges(successor, graph))) {
        auto predecessor = source(inedge, graph);
        // Ignore our own node, because it is not marked as examined yet
        if (predecessor == nf) {
          continue;
        }
        if (UEG.find(predecessor) != UEG.end()) {
          allPredecessorsExamined = false;
          break;
        }
      }
      if (allPredecessorsExamined) {
        FL.push(successor);
        llvm::outs() << "Node " << successor << " is now free\n";
      }
    }

    // Mark nf as examined
    UEG.erase(nf);
    EG.insert(nf);
  }
}

// /// Implementation of the Dominant Sequence Clustering algorithm.
// /// Described in Figure 7 of "DSC: Scheduling Parallel Tasks on an Unbounded Number of Processors"
// /// cluster is a vector that maps from a vertex to its cluster index
// void DSC(SPNGraph &graph, std::vector<cluster_t> cluster) {
//   // List of examined nodes
//   std::vector<SPNGraph::vertex_descriptor> EG(num_vertices(graph));
//   EG.reserve(num_vertices(graph));

//   // List of unexamined nodes
//   std::vector<SPNGraph::vertex_descriptor> UEG;
//   UEG.reserve(num_vertices(graph));

//   // Mark all nodes as unexamined
//   for (auto v : boost::make_iterator_range(vertices(graph))) {
//     UEG.push_back(v);
//   }

//   // List of free nodes
//   std::vector<SPNGraph::vertex_descriptor> FL;

//   // List of partially free nodes
//   std::vector<SPNGraph::vertex_descriptor> PFL;

//   // Add all free nodes to FL. A node is free if all of its predecessors are examined.
//   // At the beginning, all entry nodes are free.
//   for (auto v : boost::make_iterator_range(vertices(graph))) {
//     if (boost::in_degree(v, graph) == 0) {
//       FL.push_back(v);
//     }
//   }

//   // Calculate blevel
//   std::vector<level_t> blevel(num_vertices(graph));
//   // TODO: Compute blevel for every node

//   // Initialize with tlevel=0 for every node
//   std::vector<level_t> tlevel(num_vertices(graph), 0);

//   while (!UEG.empty()) {
//     auto nx = FL.back();  // the free task with the highest PRIO
//     auto ny = PFL.back(); // the partial free task with the highest PRIO
//   }
// }

template <typename GraphT>
Schedule<GraphT> DominantSequenceClusteringScheduler<GraphT>::operator()(GraphT &graph) {
  std::vector<cluster_t> cluster;
  Schedule<GraphT> schedule(graph);

  // Execute the Dominant Sequence Clustering algorithm
  DSC_I(graph, cluster, schedule);

  return schedule;
}

void DominantSequenceClusteringPartitioner::operator()(SPNGraph &graph) {
  std::vector<cluster_t> clusters;
  Schedule<SPNGraph> schedule(graph);

  // Execute the Dominant Sequence Clustering algorithm
  DSC_I(graph, clusters, schedule);

  // Perform the partitioning of the graph

  // A map from the cluster index to the subgraph
  // Note that cluster indices calculated by DSC_I are not contiguous!
  std::unordered_map<cluster_t, SPNGraph *> graphCluster;
  for (auto vertex : boost::make_iterator_range(vertices(graph))) {
    if (ignore_for_clustering(vertex, graph)) {
      continue;
    }

    auto cluster = clusters[vertex];

    auto it = graphCluster.find(cluster);
    if (it == graphCluster.end()) {
      // Create new subgraph for the cluster if it doesn't exist yet
      auto &subgraph = add_cluster(graph);
      boost::add_vertex(vertex, subgraph);

      graphCluster[cluster] = &subgraph;
    } else {
      // Add the vertex to the existing subgraph
      boost::add_vertex(vertex, *it->second);
    }
  }
}

// Explicit template instantiation
namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
template class DominantSequenceClusteringScheduler<SPNGraph>;
template class DominantSequenceClusteringScheduler<BSPGraph>;
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir