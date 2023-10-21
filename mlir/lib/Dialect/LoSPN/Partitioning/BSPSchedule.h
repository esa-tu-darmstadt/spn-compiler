//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "llvm/Support/GraphWriter.h"

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {

typedef unsigned int task_index_t;
typedef unsigned int processor_t;
typedef unsigned int superstep_index_t;

/// A property describing the superstep index of a BSP subgraph
struct BSPGraph_Superstep {
  using kind = boost::graph_property_tag;
};

/// A property describing the task index of a BSP vertex
struct BSPVertex_TaskID {
  using kind = boost::vertex_property_tag;
};

/// A property describing the processor index of a BSP vertex
struct BSPVertex_ProcID {
  using kind = boost::vertex_property_tag;
};

using GraphvizAttributes = std::unordered_map<std::string, std::string>;

using BSPVertexAttributes = boost::property<boost::vertex_attribute_t, GraphvizAttributes>;
using BSPVertexProperties = boost::property<BSPVertex_TaskID, task_index_t, boost::property<BSPVertex_ProcID, processor_t,  BSPVertexAttributes>>;

using BSPEdgeAttributes = boost::property<boost::edge_attribute_t, GraphvizAttributes>;
using BSPEdgeProperties =
    boost::property<boost::edge_index_t, int, BSPEdgeAttributes>;

using BSPGraphAttributes =
    boost::property<boost::graph_graph_attribute_t, GraphvizAttributes,
                    boost::property<boost::graph_vertex_attribute_t, GraphvizAttributes,
                                    boost::property<boost::graph_edge_attribute_t, GraphvizAttributes>>>;
using BSPGraphProperties =
    boost::property<boost::graph_name_t, std::string, boost::property<BSPGraph_Superstep, int, BSPGraphAttributes>>;

/// A graph that can be used to represent an SPN or a part of it.
/// Subgraphs are used to represent clusters and vertices are used to represent operations.
typedef boost::subgraph<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, BSPVertexProperties,
                                              BSPEdgeProperties, BSPGraphProperties>>
    BSPGraph;

inline void view_bspgraph(BSPGraph &graph, std::string title ) {
  // Set the vertex attributes
  for (auto vertex : boost::make_iterator_range(boost::vertices(graph))) {

    GraphvizAttributes attributes;
    processor_t proc = boost::get(BSPVertex_ProcID(), graph, vertex);
    task_index_t task = boost::get(BSPVertex_TaskID(), graph, vertex);
    attributes["label"] = "Task " + std::to_string(task) + "\nProc " + std::to_string(proc);
    attributes["shape"] = "box";
    attributes["style"] = "filled";
    attributes["fillcolor"] = "white";
    attributes["color"] = "black";

    boost::put(boost::vertex_attribute_t(), graph, vertex, attributes);
  }

  // Set cluster attributes
  for (auto &cluster : boost::make_iterator_range(graph.children())) {
    GraphvizAttributes attributes;
    auto ID = boost::get_property(cluster, BSPGraph_Superstep());
    attributes["label"] = "Superstep " + std::to_string(ID);
    attributes["style"] = "filled";
    attributes["fillcolor"] = "lightgrey";

    boost::get_property(cluster, boost::graph_graph_attribute) = attributes;
    boost::get_property(cluster, boost::graph_name) = "cluster" + std::to_string(ID);
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


/// Represents a superstep in the BSP schedule. A superstep is a set of tasks that can be executed in parallel.
class Superstep {
public:
  typedef std::shared_ptr<Superstep> Reference;

  /// Returns the predecessors of this superstep.
  auto &predecessors() const { return predecessors_; }

  /// Returns the successors of this superstep.
  auto &successors() const { return successors_; }

  /// Returns the tasks and their assigned processors of this superstep.
  auto &tasks() const { return tasks_; }

  /// Returns or sets the processor of the given task.
  processor_t &operator[](task_index_t task) { return tasks_[task]; }

private:
  std::set<Reference> predecessors_;
  std::set<Reference> successors_;
  std::unordered_map<task_index_t, processor_t> tasks_;
};

/// Represents a BSP schedule. The Bulk Synchronous Parallel (BSP) model is a parallel programming model in which
/// computation is divided into supersteps. Supersteps consists of three phases: computation, communication, and
/// synchronization. All supersteps run synchronous on all processors, ie., they begin at the same time and
/// communication between processors is only possible inbetween supersteps.
class BSPSchedule {
public:
  /// Constructs a new BSP schedule with the given number of empty supersteps.
  explicit BSPSchedule(size_t numSupersteps) : supersteps_(numSupersteps) {}

  /// Returns the supersteps of this schedule.
  auto &supersteps() const { return supersteps_; }

private:
  std::vector<Superstep> supersteps_;
};
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir