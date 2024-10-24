//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H

#include "BSPSchedule.h"
#include "SPNGraph.h"
#include "TargetExecutionModel.h"

#include <boost/fusion/algorithm/transformation/flatten.hpp>
#include <boost/range.hpp>
#include <boost/range/adaptors.hpp>

#include <boost/range/algorithm/merge.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/range/join.hpp>
#include <unordered_map>

namespace mlir {
class PatternRewriter;
namespace spn {
namespace low {

namespace partitioning {
class BSPSchedule;
template <class GraphT>
class Schedule;

class GraphPartitioner {
  SPNGraph graph_;
  const TargetExecutionModel &targetModel_;

  /// Creates a BSP graph in which vertices represent a clusters of the SPN
  /// graph.
  void createBSPGraphFromClusteredSPNGraph(SPNGraph &spnGraph,
                                           BSPGraph &bspGraph);

public:
  explicit GraphPartitioner(llvm::ArrayRef<mlir::Operation *> rootNodes,
                            const TargetExecutionModel &targetModel,
                            size_t maxTaskSize);

  /// Clusters the SPN graph
  void clusterGraph();

  /// Schedules the clustered SPN graph
  BSPSchedule scheduleGraphForBSP();

  /// Clones constants that are used by multiple clusters.
  /// If a constant has a use in a different clusters, clone the constant to the
  /// other cluster to avoid unnecessary edges crossing partitions.
  void postprocessConstants(PatternRewriter &rewriter);

  /// Returns a range of all clusters
  auto clusters() { return boost::make_iterator_range(graph_.children()); }

  /// Returns the underlying graph
  SPNGraph &graph() { return graph_; }

  /// Returns the number of clusters
  std::size_t numClusters() const { return graph_.num_children(); }

  /// Returns the maximum cluster size
  unsigned getMaximumClusterSize() const;

  /// Returns a vector of all global edges going into the cluster
  std::vector<SPNGraph::edge_descriptor> edges_in(SPNGraph &cluster) {
    std::vector<SPNGraph::edge_descriptor> inedges;

    // Iterate through all vertices of the cluster
    for (auto vertexTo : boost::make_iterator_range(boost::vertices(cluster))) {
      auto globalVertexTo = cluster.local_to_global(vertexTo);

      // Iterate through global in edges of the vertex
      for (auto globalEdge : boost::make_iterator_range(
               boost::in_edges(globalVertexTo, graph_))) {
        // Check whether the cluster contains the edge
        auto partitionHasEdge = cluster.find_edge(globalEdge);
        // If not, it is an in edge of the cluster
        if (!partitionHasEdge.second)
          inedges.push_back(globalEdge);
      }
    }

    return inedges;
  }

  /// Returns a vector of all global edges going out of the cluster
  std::vector<SPNGraph::edge_descriptor> edges_out(SPNGraph &cluster) {
    std::vector<SPNGraph::edge_descriptor> outedges;

    // Iterate through all vertices of the cluster
    for (auto vertexFrom :
         boost::make_iterator_range(boost::vertices(cluster))) {
      auto globalVertexFrom = cluster.local_to_global(vertexFrom);

      // Iterate through global out edges of the vertex
      for (auto globalEdge : boost::make_iterator_range(
               boost::out_edges(globalVertexFrom, graph_))) {
        // Check whether the cluster contains the edge
        auto partitionHasEdge = cluster.find_edge(globalEdge);
        // If not, it is an out edge of the cluster
        if (!partitionHasEdge.second)
          outedges.push_back(globalEdge);
      }
    }

    return outedges;
  }

  class ClusteringAlgorithm {
  protected:
    size_t maxClusterSize_;
    const TargetExecutionModel &targetModel_;

  public:
    ClusteringAlgorithm(const TargetExecutionModel &targetModel,
                        size_t maxClusterSize)
        : maxClusterSize_(maxClusterSize), targetModel_(targetModel) {}
    virtual ~ClusteringAlgorithm() = default;
    virtual void operator()(SPNGraph &graph) = 0;
  };

  template <class GraphT>
  class SchedulingAlgorithm {
  protected:
    const TargetExecutionModel &targetModel_;

  public:
    SchedulingAlgorithm(const TargetExecutionModel &targetModel)
        : targetModel_(targetModel) {}
    virtual ~SchedulingAlgorithm() = default;
    virtual Schedule<GraphT> operator()(GraphT &graph) = 0;
  };

protected:
  size_t maxPartitionSize_;
};

} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
