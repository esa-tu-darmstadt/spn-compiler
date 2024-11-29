//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "GraphPartitioner.h"
#include "LoSPN/LoSPNOps.h"
#include "SPNGraph.h"
#include "Schedule.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
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

SPNGraph::vertex_descriptor add_vertex_recursive(
    SPNGraph &graph, Operation *op,
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

GraphPartitioner::GraphPartitioner(SPNBody body,
                                   const TargetExecutionModel &targetModel,
                                   size_t maxTaskSize)
    : graph_(), targetModel_(targetModel), maxPartitionSize_{maxTaskSize} {
  std::unordered_map<Operation *, SPNGraph::vertex_descriptor> mapping;

  body.walk([&](SPNYield yield) {
    add_vertex_recursive(graph_, yield, mapping, targetModel);
  });
}

unsigned int GraphPartitioner::getMaximumClusterSize() const {
  // Allow up to 1% or at least one node in slack.
  unsigned slack = std::max(
      1u, static_cast<unsigned>(static_cast<double>(maxPartitionSize_) * 0.01));
  return maxPartitionSize_ + slack;
}

void GraphPartitioner::clusterGraph() {
  std::unique_ptr<TopologicalSortClustering> cluster =
      std::make_unique<TopologicalSortClustering>(targetModel_,
                                                  maxPartitionSize_);
  (*cluster)(graph_);
  view_spngraph(graph_, "Topological sort clustering");
}

SchedulingGraph
GraphPartitioner::createBSPGraphFromClusteredSPNGraph(SPNGraph &spnGraph) {
  SchedulingGraph bspGraph;
  // Maps clusters in the SPN graph to vertices in the BSP graph
  std::unordered_map<SPNGraph *, SchedulingGraph::vertex_descriptor>
      clusterToVertex;

  // Add a vertex for each cluster
  for (auto &cluster : clusters()) {
    auto vertex = add_vertex(bspGraph);
    auto clusterIndex = boost::get_property(cluster, SPNGraph_ClusterID());
    boost::put(SchedulingVertex_ClusterID(), bspGraph, vertex, clusterIndex);

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
      bool edgeExists = false;
      for (auto edge : boost::make_iterator_range(
               boost::out_edges(predecessorVertex, bspGraph))) {
        if (boost::target(edge, bspGraph) == successorVertex) {
          // Edge already exists, add the weight
          auto currentWeight = boost::get(edge_weight(), bspGraph, edge);
          boost::put(edge_weight(), bspGraph, edge, currentWeight + edgeWeight);
          edgeExists = true;
          continue;
        }
      }

      if (edgeExists)
        continue;

      // Create a new edge with the given weight
      auto edge = add_edge(predecessorVertex, successorVertex, bspGraph);
      boost::put(edge_weight(), bspGraph, edge.first, edgeWeight);
    }
  }
  return bspGraph;
}

BSPSchedule GraphPartitioner::scheduleGraphForBSP() {
  // Create the BSP graph without subgraphs / supersteps first, then assign
  // clusters to supersteps later.

  SchedulingGraph bspGraph = createBSPGraphFromClusteredSPNGraph(graph_);

  // Schedule the BSP graph
  auto scheduler_dsc =
      std::make_unique<DominantSequenceClusteringScheduler>(targetModel_);
  Schedule schedule = (*scheduler_dsc)(std::move(bspGraph));
  schedule.updateGraph();

  // schedule.viewSchedule(targetModel_,
  //                       "Dominant sequence clustering async schedule",
  //                       "/workspaces/spn/schedule_async.html");

  BSPSchedule bspSchedule = BSPSchedule::fromSchedule(std::move(schedule));
  bspSchedule.clusterGraph();

  view_schedulinggraph(bspSchedule.graph(), "BSP graph (scheduled)");

  // bspSchedule.viewSchedule(targetModel_,
  //                          "Dominant sequence clustering BSP schedule",
  //                          "/workspaces/spn/schedule_bsp.html");

  return bspSchedule;
}

void GraphPartitioner::postprocessConstants(PatternRewriter &rewriter) {
  for (auto &cluster : this->clusters()) {
    for (auto globalOutEdge : this->edges_out(cluster)) {
      auto globalVertexFrom = boost::source(globalOutEdge, this->graph());
      if (boost::get(SPNVertex_IsConstant(), this->graph(), globalVertexFrom)) {
        assert(
            boost::get(SPNVertex_Operation(), this->graph(), globalVertexFrom)
                ->getNumResults() == 1);
        // This constant is used by another partition.
        // Find the partition that uses the constant
        auto globalVertexTo = boost::target(globalOutEdge, this->graph());
        auto &otherPart = find_cluster(globalVertexTo, this->graph());

        // Clone the constant right before the using operation and add it to
        // the same partition.
        auto restore = rewriter.saveInsertionPoint();

        // Get the constant operation and the operation thats using it
        Value value = boost::get(SPNEdge_Value(), this->graph(), globalOutEdge);
        Operation *constOperation =
            boost::get(SPNVertex_Operation(), this->graph(), globalVertexFrom);
        Operation *usingOperation =
            boost::get(SPNVertex_Operation(), this->graph(), globalVertexTo);

        rewriter.setInsertionPoint(usingOperation);
        Operation *clonedOut = rewriter.clone(*constOperation);

        // Add the cloned constant to the partition
        auto globalClonedConstant =
            add_vertex(otherPart, clonedOut, targetModel_);
        auto localClonedConstant =
            boost::add_vertex(globalClonedConstant, otherPart);

        // Add the edge from the cloned constant to the using operation in the
        // other partition
        auto localVertexTo = otherPart.global_to_local(globalVertexTo);
        add_edge(localClonedConstant, localVertexTo, otherPart,
                 clonedOut->getResult(0), targetModel_);

        usingOperation->replaceUsesOfWith(value, clonedOut->getResult(0));
        rewriter.restoreInsertionPoint(restore);

        // Remove the edge from the original constant to the using operation
        // boost::remove_edge(globalOutEdge, this->graph());
      }
    }
  }
}