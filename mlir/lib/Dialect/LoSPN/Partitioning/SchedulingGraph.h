//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================
#pragma once

#include "SPNGraph.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/pending/property.hpp>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "llvm/Support/GraphWriter.h"

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {

typedef unsigned int cluster_index_t;
typedef unsigned int processor_t;
typedef unsigned int superstep_index_t;

/// A property describing the superstep index of a cluster in the scheduling
/// graph
struct SchedulingGraph_Superstep {
  using kind = boost::graph_property_tag;
};

/// A property describing the cluster index of a vertex in the scheduling graph
struct SchedulingVertex_ClusterID {
  using kind = boost::vertex_property_tag;
};

// A property describing the start time of a vertex in the scheduling graph
struct SchedulingVertex_StartTime {
  using kind = boost::vertex_property_tag;
};

// A property describing the end time of a vertex in the scheduling graph
struct SchedulingVertex_EndTime {
  using kind = boost::vertex_property_tag;
};

/// A property describing the processor index of vertex in the scheduling graph
struct SchedulingVertex_ProcID {
  using kind = boost::vertex_property_tag;
};

using GraphvizAttributes = std::unordered_map<std::string, std::string>;

using SchedulingVertexAttributes =
    boost::property<boost::vertex_attribute_t, GraphvizAttributes>;
using SchedulingVertexProperties = boost::property<
    SchedulingVertex_ClusterID, cluster_index_t,
    boost::property<
        SchedulingVertex_StartTime, int,
        boost::property<
            SchedulingVertex_EndTime, int,
            boost::property<SchedulingVertex_ProcID, processor_t,
                            boost::property<vertex_weight, int,
                                            SchedulingVertexAttributes>>>>>;

using SchedulingEdgeAttributes =
    boost::property<boost::edge_attribute_t, GraphvizAttributes>;
using SchedulingEdgeProperties = boost::property<
    boost::edge_index_t, int,
    boost::property<edge_weight, int, SchedulingEdgeAttributes>>;

using SchedulingGraphAttributes = boost::property<
    boost::graph_graph_attribute_t, GraphvizAttributes,
    boost::property<
        boost::graph_vertex_attribute_t, GraphvizAttributes,
        boost::property<boost::graph_edge_attribute_t, GraphvizAttributes>>>;
using SchedulingGraphProperties = boost::property<
    boost::graph_name_t, std::string,
    boost::property<SchedulingGraph_Superstep, superstep_index_t,
                    SchedulingGraphAttributes>>;

/// A graph that can be used to represent an SPN or a part of it.
/// Subgraphs are used to represent clusters and vertices are used to represent
/// operations.
typedef boost::subgraph<boost::adjacency_list<
    boost::vecS, boost::vecS, boost::bidirectionalS, SchedulingVertexProperties,
    SchedulingEdgeProperties, SchedulingGraphProperties>>
    SchedulingGraph;

inline void view_schedulinggraph(SchedulingGraph &graph, std::string title) {
  // Set the vertex attributes
  for (auto vertex : boost::make_iterator_range(boost::vertices(graph))) {

    GraphvizAttributes attributes;
    processor_t proc = boost::get(SchedulingVertex_ProcID(), graph, vertex);
    cluster_index_t cluster =
        boost::get(SchedulingVertex_ClusterID(), graph, vertex);
    auto weight = boost::get(vertex_weight(), graph, vertex);
    attributes["label"] =
        "Cluster " + std::to_string(cluster) + "\nProc " +
        std::to_string(proc) + "\nWeight " + std::to_string(weight) +
        "\nStart " +
        std::to_string(
            boost::get(SchedulingVertex_StartTime(), graph, vertex)) +
        "\nEnd " +
        std::to_string(boost::get(SchedulingVertex_EndTime(), graph, vertex));
    attributes["shape"] = "box";
    attributes["style"] = "filled";
    attributes["fillcolor"] = "white";
    attributes["color"] = "black";

    boost::put(boost::vertex_attribute_t(), graph, vertex, attributes);
  }

  // Set edge attributes
  for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
    auto weight = boost::get(edge_weight(), graph, edge);

    GraphvizAttributes attributes;
    attributes["label"] = std::to_string(weight);

    boost::put(boost::edge_attribute_t(), graph, edge, attributes);
  }

  // Set cluster attributes
  for (auto &cluster : boost::make_iterator_range(graph.children())) {
    GraphvizAttributes attributes;
    auto ID = boost::get_property(cluster, SchedulingGraph_Superstep());
    attributes["label"] = "Superstep " + std::to_string(ID);
    attributes["style"] = "filled";
    attributes["fillcolor"] = "lightgrey";

    boost::get_property(cluster, boost::graph_graph_attribute) = attributes;
    boost::get_property(cluster, boost::graph_name) =
        "cluster" + std::to_string(ID);
  }

  boost::get_property(graph, boost::graph_name) = "";
  boost::get_property(graph, boost::graph_graph_attribute)["label"] = title;

  // Create a temporary file to hold the graph.
  int FD;
  auto fileName = llvm::createGraphFilename("partitioning", FD);
  if (fileName.empty()) {
    return;
  }

  // Write the graph.
  boost::write_graphviz(fileName, graph);

  // Display the graph.
  llvm::DisplayGraph(fileName, false, llvm::GraphProgram::DOT);
}

inline std::string get_label(const SchedulingGraph &g,
                             SchedulingGraph::vertex_descriptor v) {
  auto cluster = boost::get(SchedulingVertex_ClusterID(), g, v);
  return "Cluster " + std::to_string(cluster);
}

} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir