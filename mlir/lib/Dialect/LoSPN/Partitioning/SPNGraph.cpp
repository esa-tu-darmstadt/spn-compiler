#include "SPNGraph.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/range/iterator_range_core.hpp>

using namespace llvm;
using namespace mlir;
using namespace mlir::spn::low;
using namespace mlir::spn::low::partitioning;

void mlir::spn::low::partitioning::view_spngraph(SPNGraph &graph, std::string title ) {
  // Set the vertex attributes
  for (auto vertex : boost::make_iterator_range(boost::vertices(graph))) {
    auto op = boost::get(SPNVertex_Operation(), graph, vertex);
    auto isConstant = boost::get(SPNVertex_IsConstant(), graph, vertex);
    auto usesInput = boost::get(SPNVertex_UsesInput(), graph, vertex);
    auto isYield = boost::get(SPNVertex_IsYield(), graph, vertex);

    GraphvizAttributes attributes;
    attributes["label"] = op->getName().getStringRef().str();
    attributes["shape"] = "box";
    attributes["style"] = "filled";
    attributes["fillcolor"] =
        isConstant ? "lightsalmon" : (isYield ? "lightcoral" : (usesInput ? "lightblue" : "white"));
    attributes["color"] = "black";

    boost::put(boost::vertex_attribute_t(), graph, vertex, attributes);
  }

  // Set cluster attributes
  for (auto &cluster : boost::make_iterator_range(graph.children())) {
    GraphvizAttributes attributes;
    auto ID = boost::get_property(cluster, SPNGraph_ClusterID());
    attributes["label"] = "Cluster " + std::to_string(ID);
    attributes["style"] = "filled";
    attributes["fillcolor"] = "lightgrey";

    boost::get_property(cluster, boost::graph_graph_attribute) = attributes;
    boost::get_property(cluster, boost::graph_name) = "cluster" + std::to_string(ID);
  }

  // Set edge attributes
  // for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
  //   auto value = boost::get(SPNEdge_Value(), graph, edge);

  //   GraphvizAttributes attributes;
  //   llvm::raw_string_ostream edgeStream(attributes["label"]);
  //   if (value)
  //     edgeStream << "Val: " << value;
  //   else
  //     edgeStream << "Missing value!";

  //   boost::put(boost::edge_attribute_t(), graph, edge, attributes);
  // }

  boost::get_property(graph, boost::graph_name) = "";
  boost::get_property(graph, boost::graph_graph_attribute)["label"] = title;

  // Create a temporary file to hold the graph.
  int FD;
  auto fileName = createGraphFilename("partitioning", FD);
  if (fileName.empty()) {
    return;
  }

  // Write the graph.
  boost::write_graphviz(fileName, graph);

  // Display the graph.
  DisplayGraph(fileName, false, GraphProgram::DOT);
}

SPNGraph::edge_descriptor mlir::spn::low::partitioning::add_edge(SPNGraph::vertex_descriptor u,
                                                                 SPNGraph::vertex_descriptor v, SPNGraph &graph,
                                                                 Value value) {
  auto edge = boost::add_edge(u, v, graph);
  assert(edge.second && "Cannot add edge");

  boost::put(SPNEdge_Value(), graph, edge.first, value);
  return edge.first;
}

SPNGraph::vertex_descriptor mlir::spn::low::partitioning::add_vertex(SPNGraph &graph, Operation *op) {
  auto v = boost::add_vertex(graph);

  boost::put(SPNVertex_Operation(), graph, v, op);
  boost::put(SPNVertex_IsConstant(), graph, v, op->hasTrait<OpTrait::ConstantLike>());
  boost::put(SPNVertex_IsYield(), graph, v, isa<SPNYield>(op));
  bool usesInput = false;

  for (auto result : op->getOperands()) {
    usesInput |= result.isa<mlir::BlockArgument>();
  }

  boost::put(SPNVertex_UsesInput(), graph, v, usesInput);
  return v;
}

void boost::throw_exception(std::exception const &e) {
  outs() << "Exception: " << e.what() << "\n";

  exit(1);
}

void boost::throw_exception(std::exception const &e, boost::source_location const &loc) {
  outs() << "Exception at " << loc.file_name() << ":" << loc.line() << ":" << loc.column() << ": " << e.what() << "\n";

  exit(1);
}

bool mlir::spn::low::partitioning::ignore_for_clustering(SPNGraph::vertex_descriptor v, SPNGraph &graph) {
  auto op = boost::get(SPNVertex_Operation(), graph, v);
  return isa<SPNYield>(op);
}

SPNGraph &mlir::spn::low::partitioning::add_cluster(SPNGraph &graph) {
  auto &cluster = graph.create_subgraph();
  boost::get_property(cluster, SPNGraph_ClusterID()) = graph.num_children();
  return cluster;
}

SPNGraph &mlir::spn::low::partitioning::find_cluster(SPNGraph::vertex_descriptor globalVertex, SPNGraph &graph) {
  for (auto &cluster : boost::make_iterator_range(graph.children())) {
    if (cluster.find_vertex(globalVertex).second)
      return cluster;
  }

  assert(false && "Vertex not found in any cluster");
}