//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <fstream>
#include <spdlog/spdlog.h>
#include "GraphStatVisitor.h"

using namespace spnc;

GraphStatVisitor::GraphStatVisitor(ActionWithOutput<IRGraph>& _input, StatsFile outputFile)
    : ActionSingleInput<IRGraph, StatsFile>{_input}, outfile{std::move(outputFile)} {}

void GraphStatVisitor::collectGraphStats(const NodeReference rootNode) {
  std::vector<NODETYPE> nodetype_inner = {NODETYPE::SUM, NODETYPE::PRODUCT};
  std::vector<NODETYPE> nodetype_leaf = {NODETYPE::HISTOGRAM};

  spn_node_stats = {{NODETYPE::SUM, std::multimap<int, std::string>()},
                    {NODETYPE::PRODUCT, std::multimap<int, std::string>()},
                    {NODETYPE::HISTOGRAM, std::multimap<int, std::string>()}};

  std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({1}));
  rootNode->accept(*this, passed_arg);

  int count_node_temp = 0;

  for (NODETYPE n : nodetype_inner) {
    count_node_temp = spn_node_stats.find(n)->second.size();
    count_nodes_inner += count_node_temp;

    switch (n) {
      case NODETYPE::SUM:count_nodes_sum = count_node_temp;
        break;
      case NODETYPE::PRODUCT:count_nodes_product = count_node_temp;
        break;
      default:
        // Encountered specified but unhandled NODETYPE
        assert(false);
    }

  }

  for (NODETYPE n : nodetype_leaf) {
    auto nodes = spn_node_stats.find(n)->second;
    count_node_temp = nodes.size();
    count_nodes_leaf += count_node_temp;

    switch (n) {
      case NODETYPE::HISTOGRAM:count_nodes_histogram = count_node_temp;
        break;
      default:
        // Encountered specified but unhandled NODETYPE
        assert(false);
    }

    // Note: end() will (unlike begin()) point "behind" the data we're looking for, hence the decrement.
    depth_min = nodes.begin()->first;
    depth_max = (--nodes.end())->first;

    // Since the used multimap is ordered, we can simply use the respective node count to get the median index.
    int median_index_temp = count_node_temp / 2;

    // ToDo: Determining the median depth has to be revisited once other leaf nodes are supported (2020-JAN-31).
    int level = 0;
    for (auto& node : nodes) {
      level = node.first;
      depth_average += level;

      if (median_index_temp > 0) {
        --median_index_temp;
        if (median_index_temp == 0) {
          depth_median = level;
        }
      }
    }

  }

  depth_average = depth_average / count_nodes_leaf;

  SPDLOG_INFO("====================================");
  SPDLOG_INFO("|          SPN Statistics          |");
  SPDLOG_INFO("====================================");
  SPDLOG_INFO(" > Number of features: {}{}", count_features);
  SPDLOG_INFO(" > Minimum depth: {}", depth_min);
  SPDLOG_INFO(" > Maximum depth: {}", depth_max);
  SPDLOG_INFO(" > Average depth: {}", depth_average);
  SPDLOG_INFO(" > Median depth:  {}", depth_median);
  SPDLOG_INFO(" > Nodes (inner, leaf): ({}, {})", count_nodes_inner, count_nodes_leaf);
  SPDLOG_INFO(" > Sum-Nodes: {}", count_nodes_sum);
  SPDLOG_INFO(" > Product-Nodes: {}", count_nodes_product);
  SPDLOG_INFO(" > Histogram-Nodes: {}", count_nodes_histogram);
  SPDLOG_INFO("====================================");

  json stats;

  stats["count_features"] = count_features;
  stats["depth_min"] = depth_min;
  stats["depth_max"] = depth_max;
  stats["depth_average"] = depth_average;
  stats["depth_median"] = depth_median;
  stats["count_nodes_inner"] = count_nodes_inner;
  stats["count_nodes_leaf"] = count_nodes_leaf;
  stats["count_nodes_sum"] = count_nodes_sum;
  stats["count_nodes_product"] = count_nodes_product;
  stats["count_nodes_histogram"] = count_nodes_histogram;

  std::ofstream fileStream;
  fileStream.open(outfile.fileName());
  fileStream << stats;
  fileStream.close();
}

void GraphStatVisitor::visitInputvar(InputVar& n, arg_t arg) {}

void GraphStatVisitor::visitHistogram(Histogram& n, arg_t arg) {
  int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;
  spn_node_stats.find(NODETYPE::HISTOGRAM)->second.insert(std::pair<int, std::string>(currentLevel, n.id()));

  n.indexVar().accept(*this, nullptr);
}

void GraphStatVisitor::visitProduct(Product& n, arg_t arg) {
  int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;
  spn_node_stats.find(NODETYPE::PRODUCT)->second.insert(std::pair<int, std::string>(currentLevel, n.id()));

  for (auto& child : n.multiplicands()) {
    std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({currentLevel + 1}));
    child->accept(*this, passed_arg);
  }
}

void GraphStatVisitor::visitSum(Sum& n, arg_t arg) {
  int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;
  spn_node_stats.find(NODETYPE::SUM)->second.insert(std::pair<int, std::string>(currentLevel, n.id()));

  for (auto& child : n.addends()) {
    std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({currentLevel + 1}));
    child->accept(*this, passed_arg);
  }
}

void GraphStatVisitor::visitWeightedSum(WeightedSum& n, arg_t arg) {
  int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;
  spn_node_stats.find(NODETYPE::SUM)->second.insert(std::pair<int, std::string>(currentLevel, n.id()));

  for (auto& child : n.addends()) {
    std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({currentLevel + 1}));
    child.addend->accept(*this, passed_arg);
  }
}

StatsFile& spnc::GraphStatVisitor::execute() {
  if (!cached) {
    IRGraph graph = input.execute();
    count_features = graph.inputs().size();
    collectGraphStats(graph.rootNode());
    cached = true;
  }
  return outfile;
}
    
