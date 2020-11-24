//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "CollectGraphStatistics.h"
#include "SPN/SPNOps.h"
#include "util/json.hpp"

using namespace spnc;
using namespace mlir;
using namespace mlir::spn;
using json = nlohmann::json;

CollectGraphStatistics::CollectGraphStatistics(ActionWithOutput<mlir::ModuleOp>& _input, StatsFile _statsFile)
    : ActionSingleInput<mlir::ModuleOp, StatsFile>{_input}, statsFile{std::move(_statsFile)} {}

StatsFile& CollectGraphStatistics::execute() {
  if (!cached) {
    auto module = input.execute();
    collectStatistics(module);
    cached = true;
  }
  return statsFile;
}

void CollectGraphStatistics::collectStatistics(mlir::ModuleOp& module) {
  llvm::SmallVector<Operation*, 5> queries;
  module.walk([&queries](Operation* op) {
    if (auto query = dyn_cast<QueryInterface>(op)) {
      queries.push_back(op);
    }
  });

  if (queries.empty()) {
    SPDLOG_ERROR("Did not find any queries to analyze!");
    return;
  }

  if (queries.size() > 1) {
    // TODO: Maybe we can extend this to multiple queries in a single module in the future.
    SPDLOG_WARN("Found more than one query, will only analyze the first query!");
  }

  nodeLevel = std::make_unique<SPNNodeLevel>(queries.front());
  graphStats = std::make_unique<SPNGraphStatistics>(queries.front());

  auto featureCount = dyn_cast<QueryInterface>(queries.front()).getNumFeatures();

  auto sumCount = graphStats->getKindNodeCount<SumOp>();
  auto prodCount = graphStats->getKindNodeCount<ProductOp>();
  auto histCount = graphStats->getKindNodeCount<HistogramOp>();
  auto constCount = graphStats->getKindNodeCount<ConstantOp>();
  auto innerCount = graphStats->getInnerNodeCount();
  auto leafCount = graphStats->getLeafNodeCount();

  auto maxDepth = nodeLevel->getMaxDepth();
  auto minDepth = nodeLevel->getMinDepth();
  auto medianDepth = nodeLevel->getMedianDepth();
  auto avgDepth = nodeLevel->getAverageDepth();

  SPDLOG_INFO("====================================");
  SPDLOG_INFO("|          SPN Statistics          |");
  SPDLOG_INFO("====================================");
  SPDLOG_INFO(" > Number of features: {}", featureCount);
  SPDLOG_INFO(" > Minimum depth: {}", minDepth);
  SPDLOG_INFO(" > Maximum depth: {}", maxDepth);
  SPDLOG_INFO(" > Average depth: {}", avgDepth);
  SPDLOG_INFO(" > Median depth:  {}", medianDepth);
  SPDLOG_INFO(" > Nodes (inner, leaf): ({}, {})", innerCount, leafCount);
  SPDLOG_INFO(" > Sum-Nodes: {}", sumCount);
  SPDLOG_INFO(" > Product-Nodes: {}", prodCount);
  SPDLOG_INFO(" > Histogram-Nodes: {}", histCount);
  SPDLOG_INFO(" > Constant-Nodes: {}", constCount);
  SPDLOG_INFO("====================================");

  json stats;

  stats["featureCount"] = featureCount;
  stats["minDepth"] = minDepth;
  stats["maxDepth"] = maxDepth;
  stats["averageDepth"] = avgDepth;
  stats["medianDepth"] = medianDepth;
  stats["innerCount"] = innerCount;
  stats["leafCount"] = leafCount;
  stats["sumCount"] = sumCount;
  stats["productCount"] = prodCount;
  stats["histogramCount"] = histCount;
  stats["constantCount"] = constCount;

  std::ofstream fileStream;
  fileStream.open(statsFile.fileName());
  fileStream << stats;
  fileStream.close();
}