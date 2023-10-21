//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "DominantSequenceClustering.h"
#include "../../SPNGraph.h"

#include <boost/graph/subgraph.hpp>
#include <unordered_map>

using namespace mlir::spn::low::partitioning;

typedef int level_t;
typedef int cluster_t;

void DSC_I(SPNGraph &graph, std::vector<cluster_t> cluster) {
  // Calculate blevel
  std::vector<level_t> blevel(num_vertices(graph));
  // TODO: Compute blevel for every node

  // Initialize with tlevel=0 for every node
  std::vector<level_t> tlevel(num_vertices(graph), 0);

  cluster.reserve(num_vertices(graph));

  // Initialize every node into its own cluster
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    cluster[v] = v;
  }

  // List of examined nodes
  std::vector<SPNGraph::vertex_descriptor> EG(num_vertices(graph));
  EG.reserve(num_vertices(graph));

  // List of unexamined nodes
  std::vector<SPNGraph::vertex_descriptor> UEG;
  UEG.reserve(num_vertices(graph));

  // Mark all nodes as unexamined
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    UEG.push_back(v);
  }

  // While there are unexamined nodes
  while (!UEG.empty()) {
    // TODO: Find a free node with highest priority from UEG
    SPNGraph::vertex_descriptor nf = 0;

    // Merge nf with the cluster of one of its predecessors such that tlevel(nf) decreases in a maximal way.
    // If all zeroing increase tlevel(nf), nf remains in its own cluster.

    // Captures the best predecessor to merge with and the new tlevel of nf
    SPNGraph::vertex_descriptor predecessorToMergeWith = SPNGraph::null_vertex();
    level_t bestTLevel = tlevel[nf];

    for (auto inedge : boost::make_iterator_range(in_edges(nf, graph))) {
      auto predecessor = source(inedge, graph);
      level_t newTLevel = 0; // FIXME tlevel(nf, predecessor);
      if (newTLevel < bestTLevel) {
        bestTLevel = newTLevel;
        predecessorToMergeWith = predecessor;
      }
    }

    // Merge nf with the cluster of predecessorToMergeWith if we found a predecessor to merge with
    if (predecessorToMergeWith != SPNGraph::null_vertex()) {
      cluster[nf] = cluster[predecessorToMergeWith];
    }

    // TODO: Update the priorities of nf's successors

    // TODO: Mark nf as examined
  }
}

void DominantSequenceClustering::operator()(SPNGraph &graph) {
  // Maps from a vertex to its cluster index
  std::vector<cluster_t> clusters;

  // Calculate the clustering
  DSC_I(graph, clusters);

  // Perform the clustering of the actual graph

  // A map from the cluster index to the subgraph
  // Note that cluster indices calculated by DSC_I are not contiguous!
  std::unordered_map<cluster_t, SPNGraph *> graphCluster;
  for (auto vertex : boost::make_iterator_range(vertices(graph))) {
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