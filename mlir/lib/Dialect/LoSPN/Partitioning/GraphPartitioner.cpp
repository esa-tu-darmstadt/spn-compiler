//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "GraphPartitioner.h"
#include "SPNGraph.h"
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

SPNGraph::vertex_descriptor
add_vertex_recursive(SPNGraph &graph, Operation *op,
                     std::unordered_map<Operation *, SPNGraph::vertex_descriptor> &mapping) {
  // Check if the operation is already in the graph.
  auto it = mapping.find(op);
  if (it != mapping.end()) {
    return it->second;
  }

  // Create a new vertex.
  auto v = add_vertex(graph, op);
  mapping[op] = v;

  // Operands connect to either other operations or block arguments.
  for (Value operand : op->getOperands()) {
    if (Operation *definingOp = operand.getDefiningOp()) {
      auto u = add_vertex_recursive(graph, definingOp, mapping);
      add_edge(u, v, graph, operand);
    }
  }

  return v;
}

GraphPartitioner::GraphPartitioner(llvm::ArrayRef<mlir::Operation *> rootNodes, size_t maxTaskSize)
    : graph_(), maxPartitionSize_{maxTaskSize} {
  std::unordered_map<Operation *, SPNGraph::vertex_descriptor> mapping;

  for (auto rootNode : rootNodes) {
    add_vertex_recursive(graph_, rootNode, mapping);
  }
}

unsigned int GraphPartitioner::getMaximumClusterSize() const {
  // Allow up to 1% or at least one node in slack.
  unsigned slack = std::max(1u, static_cast<unsigned>(static_cast<double>(maxPartitionSize_) * 0.01));
  return maxPartitionSize_ + slack;
}

void GraphPartitioner::clusterGraph() {
  view_spngraph(graph_, "Before clustering");

  SPNGraph graph_topo(graph_);
  std::unique_ptr<TopologicalSortClustering> cluster = std::make_unique<TopologicalSortClustering>(maxPartitionSize_);
  (*cluster)(graph_topo);
  view_spngraph(graph_topo, "Topological sort clustering");

  // SPNGraph graph_dsc(graph_);
  // std::unique_ptr<DominantSequenceClustering> cluster_dsc =
  //     std::make_unique<DominantSequenceClustering>(maxPartitionSize_);
  // (*cluster_dsc)(graph_dsc);
  // view_spngraph(graph_dsc, "Dominant sequence clustering");

  graph_ = graph_topo;
}

BSPSchedule GraphPartitioner::scheduleGraphForBSP() {
  // Create a BSP graph in which subgraphs represent supersteps and vertices represent a tasks.
  // Tasks are clusters in the SPN graph.

  // Create the BSP graph without subgraphs / supersteps first, then assign tasks to supersteps later.

  BSPGraph bspGraph;

  // Add a vertex for each cluster
  task_index_t taskIndex = 0;
  std::unordered_map<SPNGraph *, BSPGraph::vertex_descriptor> clusterToTask;
  for (auto &cluster : clusters()) {
    auto task = add_vertex(bspGraph);
    boost::put(BSPVertex_TaskID(), bspGraph, task, taskIndex++);

    clusterToTask[&cluster] = task;
  }

  // Add edges between tasks / clusters
  for (auto &cluster : clusters()) {
    for (auto inedge : this->edges_in(cluster)) {
      auto predecessorVertex = source(inedge, graph_);
      auto predecessorCluster = find_cluster(predecessorVertex, graph_);

      auto predecessorTask = clusterToTask[&predecessorCluster];
      auto successorTask = clusterToTask[&cluster];

      add_edge(predecessorTask, successorTask, bspGraph);
    }
  }

  // Assign tasks to supersteps according to their wavefront
  std::vector<superstep_index_t> superstepOfTask(boost::num_vertices(bspGraph), 0);
  size_t numSupersteps = 0;
  for (auto task : boost::make_iterator_range(boost::vertices(bspGraph))) {
    auto superStep = boost::ith_wavefront(task, bspGraph);
    numSupersteps = std::max(numSupersteps, superStep);
    superstepOfTask[task] = superStep;
  }

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
    auto superStep = superstepOfTask[task];
    auto &superstep = *superstepSubgraphs[superStep];
    boost::add_vertex(task, superstep);
  }

  // View the graph
  view_bspgraph(bspGraph, "BSP graph");
}