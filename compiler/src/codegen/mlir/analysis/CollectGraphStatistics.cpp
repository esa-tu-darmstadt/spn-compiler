//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "CollectGraphStatistics.h"
#include "LoSPN/LoSPNOps.h"
#include "util/json.hpp"

using namespace spnc;
using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
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
  // ToDo: Do we need previous checks / assertions?

  // ToDo: Adapt SPNNodeLevel analysis to loSPN dialect.
  // nodeLevel = std::make_unique<SPNNodeLevel>(module);
  graphStats = std::make_unique<SPNGraphStatistics>(module);

  // ToDo: How to get the featureCount from loSPN?
  auto featureCount = -1; //  dyn_cast<QueryInterface>(queries.front()).getNumFeatures();

  auto sumCount = graphStats->getKindNodeCount<SPNAdd>();
  auto prodCount = graphStats->getKindNodeCount<SPNMul>();
  auto categCount = graphStats->getKindNodeCount<SPNCategoricalLeaf>();
  auto constCount = graphStats->getKindNodeCount<SPNConstant>();
  auto gaussCount = graphStats->getKindNodeCount<SPNGaussianLeaf>();
  auto histCount = graphStats->getKindNodeCount<SPNHistogramLeaf>();
  auto innerCount = graphStats->getInnerNodeCount();
  auto leafCount = graphStats->getLeafNodeCount();

  // ToDo: Adapt SPNNodeLevel analysis to loSPN dialect.
  auto maxDepth = -1; // nodeLevel->getMaxDepth();
  auto minDepth = -1; // nodeLevel->getMinDepth();
  auto medianDepth = -1; // nodeLevel->getMedianDepth();
  auto avgDepth = -1; // nodeLevel->getAverageDepth();

  SPDLOG_INFO("====================================");
  SPDLOG_INFO("|          SPN Statistics          |");
  SPDLOG_INFO("====================================");
  SPDLOG_INFO(" > Number of features: {}", featureCount);
  SPDLOG_INFO(" > Minimum depth: {}", minDepth);
  SPDLOG_INFO(" > Maximum depth: {}", maxDepth);
  SPDLOG_INFO(" > Average depth: {}", avgDepth);
  SPDLOG_INFO(" > Median depth:  {}", medianDepth);
  SPDLOG_INFO(" > Nodes (inner, leaf): ({}, {})", innerCount, leafCount);
  SPDLOG_INFO(" > Sum-Nodes:         {}", sumCount);
  SPDLOG_INFO(" > Product-Nodes:     {}", prodCount);
  SPDLOG_INFO(" > Categorical-Nodes: {}", categCount);
  SPDLOG_INFO(" > Constant-Nodes:    {}", constCount);
  SPDLOG_INFO(" > Gaussian-Nodes:    {}", gaussCount);
  SPDLOG_INFO(" > Histogram-Nodes:   {}", histCount);
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
  stats["categoricalCount"] = categCount;
  stats["constantCount"] = constCount;
  stats["gaussianCount"] = gaussCount;
  stats["histogramCount"] = histCount;

  std::ofstream fileStream;
  fileStream.open(statsFile.fileName());
  fileStream << stats;
  fileStream.close();
}