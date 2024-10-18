//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

#include "TargetExecutionModel.h"
#include "mlir/IR/Value.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/subgraph.hpp>

#include <unordered_map>

namespace mlir {
class Operation;
namespace spn {
namespace low {
class SPNYield;
namespace partitioning {

/// A property describing the underlying mlir Operation of an vertex
struct SPNVertex_Operation {
  using kind = boost::vertex_property_tag;
};

/// A property describing whether a vertex uses user input, i.e., block argument
struct SPNVertex_UsesInput {
  using kind = boost::vertex_property_tag;
};

/// A property describing whether a vertex is a constant
struct SPNVertex_IsConstant {
  using kind = boost::vertex_property_tag;
};

/// A property describing whether a vertex is a yield
struct SPNVertex_IsYield {
  using kind = boost::vertex_property_tag;
};

/// A property describing the underlying mlir Value of an edge
struct SPNEdge_Value {
  using kind = boost::edge_property_tag;
};

/// A property describing the cluster ID of a subgraph
struct SPNGraph_ClusterID {
  using kind = boost::graph_property_tag;
};

/// A property describing whether a cluster is partitioned to a task yet
struct SPNGraph_IsTaskPartitioned {
  using kind = boost::graph_property_tag;
};

/// A property describing the weight of a vertex
struct vertex_weight {
  using kind = boost::vertex_property_tag;
};

/// A property describing the weight of an edge
struct edge_weight {
  using kind = boost::edge_property_tag;
};

using GraphvizAttributes = std::unordered_map<std::string, std::string>;

using VertexAttributes = boost::property<boost::vertex_attribute_t, GraphvizAttributes>;
using VertexProperties = boost::property<
    SPNVertex_Operation, mlir::Operation *,
    boost::property<SPNVertex_IsConstant, bool,
                    boost::property<SPNVertex_UsesInput, bool,
                                    boost::property<SPNVertex_IsYield, bool,
                                                    boost::property<vertex_weight, int, VertexAttributes>>>>>;

using EdgeAttributes = boost::property<boost::edge_attribute_t, GraphvizAttributes>;
using EdgeProperties =
    boost::property<boost::edge_index_t, int,
                    boost::property<SPNEdge_Value, mlir::Value, boost::property<edge_weight, int, EdgeAttributes>>>;

using GraphAttributes =
    boost::property<boost::graph_graph_attribute_t, GraphvizAttributes,
                    boost::property<boost::graph_vertex_attribute_t, GraphvizAttributes,
                                    boost::property<boost::graph_edge_attribute_t, GraphvizAttributes>>>;
using GraphProperties =
    boost::property<boost::graph_name_t, std::string, boost::property<SPNGraph_ClusterID, int, GraphAttributes>>;

/// A graph that can be used to represent an SPN or a part of it.
/// Subgraphs are used to represent clusters and vertices are used to represent operations.
typedef boost::subgraph<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, VertexProperties,
                                              EdgeProperties, GraphProperties>>
    SPNGraph;

/// Add a vertex for the given operation to the graph.
SPNGraph::vertex_descriptor add_vertex(SPNGraph &graph, Operation *op, const TargetExecutionModel &targetModel);

/// Add a vertex for the given value to the graph.
SPNGraph::edge_descriptor add_edge(SPNGraph::vertex_descriptor u, SPNGraph::vertex_descriptor v, SPNGraph &graph,
                                   Value value, const TargetExecutionModel &targetModel);

/// Adds a cluster to the graph
SPNGraph &add_cluster(SPNGraph &graph);

/// Returns the cluster that contains the given global vertex
SPNGraph &find_cluster(SPNGraph::vertex_descriptor globalVertex, SPNGraph &graph);

/// Returns true if the given vertex should not be added to a cluster
bool ignore_for_clustering(SPNGraph::vertex_descriptor v, SPNGraph &graph);

/// View the SPN graph in the default graphviz viewer.
void view_spngraph(SPNGraph &graph, std::string title = "");

std::string get_label(const SPNGraph &g, SPNGraph::vertex_descriptor v);

} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir