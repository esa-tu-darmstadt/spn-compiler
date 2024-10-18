//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "GraphPartitioner.h"
#include "SPNGraph.h"
#include "Schedule.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/graph/wavefront.hpp>
#include <boost/pending/property.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <unordered_map>

#include "Algorithms/DominantSequenceClustering/DominantSequenceClustering.h"
#include "Algorithms/TopologicalSort/TopoSortClustering.h"

#include "BSPSchedule.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::spn::low;
using namespace mlir::spn::low::partitioning;

SPNGraph::vertex_descriptor add_vertex_recursive(SPNGraph &graph, Operation *op,
                                                 std::unordered_map<Operation *, SPNGraph::vertex_descriptor> &mapping,
                                                 const TargetExecutionModel &targetModel) {
  // Check if the operation is already in the graph.
  auto it = mapping.find(op);
  if (it != mapping.end()) {
    return it->second;
  }

  // Create a new vertex.
  auto v = add_vertex(graph, op, targetModel);
  mapping[op] = v;

  // Operands connect to either other operations or block arguments.
  for (Value operand : op->getOperands()) {
    if (Operation *definingOp = operand.getDefiningOp()) {
      auto u = add_vertex_recursive(graph, definingOp, mapping, targetModel);
      add_edge(u, v, graph, operand, targetModel);
    }
  }

  return v;
}

GraphPartitioner::GraphPartitioner(llvm::ArrayRef<mlir::Operation *> rootNodes, const TargetExecutionModel &targetModel,
                                   size_t maxTaskSize)
    : graph_(), targetModel_(targetModel), maxPartitionSize_{maxTaskSize} {
  std::unordered_map<Operation *, SPNGraph::vertex_descriptor> mapping;

  for (auto rootNode : rootNodes) {
    add_vertex_recursive(graph_, rootNode, mapping, targetModel);
  }
}

unsigned int GraphPartitioner::getMaximumClusterSize() const {
  // Allow up to 1% or at least one node in slack.
  unsigned slack = std::max(1u, static_cast<unsigned>(static_cast<double>(maxPartitionSize_) * 0.01));
  return maxPartitionSize_ + slack;
}

void GraphPartitioner::clusterGraph() {
  view_spngraph(graph_, "Before clustering");

  // SPNGraph graph_topo(graph_);
  // std::unique_ptr<TopologicalSortClustering> cluster =
  // std::make_unique<TopologicalSortClustering>(maxPartitionSize_);
  // (*cluster)(graph_);
  // view_spngraph(graph_, "Topological sort clustering");

  // SPNGraph graph_dsc(graph_);
  std::unique_ptr<DominantSequenceClusteringPartitioner> cluster_dsc =
      std::make_unique<DominantSequenceClusteringPartitioner>(targetModel_, maxPartitionSize_);
  (*cluster_dsc)(graph_);
  view_spngraph(graph_, "Dominant sequence clustering");

  // This somehow does not work
  // graph_ = graph_topo;
}

void GraphPartitioner::createBSPGraphFromClusteredSPNGraph(SPNGraph &spnGraph, BSPGraph &bspGraph) {
  // Maps clusters in the SPN graph to vertices in the BSP graph
  std::unordered_map<SPNGraph *, BSPGraph::vertex_descriptor> clusterToVertex;

  // Add a vertex for each cluster
  for (auto &cluster : clusters()) {
    auto vertex = add_vertex(bspGraph);
    auto clusterIndex = boost::get_property(cluster, SPNGraph_ClusterID());
    boost::put(BSPVertex_ClusterID(), bspGraph, vertex, clusterIndex);

    // Calculate the weight of the cluster
    int weight = 0;
    for (auto vertex : boost::make_iterator_range(boost::vertices(cluster))) {
      weight += boost::get(vertex_weight(), cluster, vertex);
    }
    boost::put(vertex_weight(), bspGraph, vertex, weight);

    clusterToVertex[&cluster] = vertex;
  }

  // Add edges between cluster vertices
  for (auto &cluster : clusters()) {
    for (auto inedge : this->edges_in(cluster)) {
      auto predecessorOp = source(inedge, spnGraph);
      auto &predecessorCluster = find_cluster(predecessorOp, spnGraph);

      auto predecessorVertex = clusterToVertex[&predecessorCluster];
      auto successorVertex = clusterToVertex[&cluster];

      int edgeWeight = boost::get(edge_weight(), spnGraph, inedge);

      // Check if the edge already exists
      for (auto edge : boost::make_iterator_range(boost::out_edges(predecessorVertex, bspGraph))) {
        if (boost::target(edge, bspGraph) == successorVertex) {
          // Edge already exists, add the weight
          auto currentWeight = boost::get(edge_weight(), bspGraph, edge);
          boost::put(edge_weight(), bspGraph, edge, currentWeight + edgeWeight);
          continue;
        }
      }

      // Create a new edge with the given weight
      auto edge = add_edge(predecessorVertex, successorVertex, bspGraph);
      boost::put(edge_weight(), bspGraph, edge.first, edgeWeight);
    }
  }
}

BSPSchedule GraphPartitioner::scheduleGraphForBSP() {
  // Create the BSP graph without subgraphs / supersteps first, then assign clusters to supersteps later.

  BSPGraph bspGraph;

  // Fill the BSP graph with a vertex for each cluster and an edge for each edge between clusters
  createBSPGraphFromClusteredSPNGraph(graph_, bspGraph);

  // Schedule the BSP graph
  std::unique_ptr<DominantSequenceClusteringScheduler<BSPGraph>> scheduler_dsc =
      std::make_unique<DominantSequenceClusteringScheduler<BSPGraph>>(targetModel_);
  Schedule<BSPGraph> schedule = (*scheduler_dsc)(bspGraph);

  schedule.viewSchedule(targetModel_);

  // Assign clusters to supersteps according to their wavefront
  std::vector<superstep_index_t> superstepOfCluster(boost::num_vertices(bspGraph), 0);
  size_t numSupersteps = 0;
  for (auto cluster : boost::make_iterator_range(boost::vertices(bspGraph))) {
    auto superStep = boost::ith_wavefront(cluster, bspGraph);
    numSupersteps = std::max(numSupersteps, superStep + 1);
    superstepOfCluster[cluster] = superStep;

    // Debug output
    auto clusterIndex = boost::get(BSPVertex_ClusterID(), bspGraph, cluster);
    llvm::errs() << "Cluster " << clusterIndex << " is in superstep " << superStep << "\n";
  }

  view_bspgraph(bspGraph, "BSP graph (unscheduled)");
  // Cluster the graph into subgraphs (for visualization)
  // Add a subgraph for each superstep
  std::vector<BSPGraph *> superstepSubgraphs;
  superstepSubgraphs.reserve(numSupersteps);
  for (size_t i = 0; i < numSupersteps; ++i) {
    BSPGraph &superstep = bspGraph.create_subgraph();
    superstepSubgraphs.push_back(&superstep);
    boost::get_property(superstep, BSPGraph_Superstep()) = i;
  }
  // Add the vertices to the subgraphs
  for (auto task : boost::make_iterator_range(boost::vertices(bspGraph))) {
    auto superStep = superstepOfCluster[task];
    auto &superstep = *superstepSubgraphs[superStep];
    boost::add_vertex(task, superstep);
  }

  // View the graph
  view_bspgraph(bspGraph, "BSP graph");
  BSPSchedule bspSchedule(bspGraph.num_children());
  return bspSchedule;
}