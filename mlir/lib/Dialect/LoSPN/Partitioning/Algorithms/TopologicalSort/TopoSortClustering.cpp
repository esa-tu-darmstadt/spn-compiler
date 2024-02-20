#include "TopoSortClustering.h"
#include "../../SPNGraph.h"

#include <boost/graph/subgraph.hpp>
#include <boost/graph/topological_sort.hpp>

using namespace mlir::spn::low::partitioning;

void TopologicalSortClustering::operator()(SPNGraph &graph) {
  std::vector<SPNGraph::vertex_descriptor> topoOrder;
  boost::topological_sort(graph, std::back_inserter(topoOrder));

  auto *cluster = &add_cluster(graph);
  while (topoOrder.size() > 0) {
    auto vertexGlobal = topoOrder.back();
    topoOrder.pop_back();
    if (ignore_for_clustering(vertexGlobal, graph))
      continue;

    if (boost::num_vertices(*cluster) >= maxClusterSize_)
      cluster = &add_cluster(graph);

    boost::add_vertex(vertexGlobal, *cluster);
  }
}