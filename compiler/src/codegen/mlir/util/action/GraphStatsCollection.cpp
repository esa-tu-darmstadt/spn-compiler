//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "GraphStatsCollection.h"

using namespace mlir;
using namespace mlir::spn;
using namespace spnc;

GraphStatsCollection::GraphStatsCollection(ActionWithOutput<ModuleOp>& _input, StatsFile outputFile)
    : ActionSingleInput<ModuleOp, StatsFile>{_input}, outfile{std::move(outputFile)} {
}

void GraphStatsCollection::collectGraphStats() {
  count_nodes_sum = statistics_count.getCountNodes(NODETYPE::SUM);
  count_nodes_product = statistics_count.getCountNodes(NODETYPE::PRODUCT);
  count_nodes_histogram = statistics_count.getCountNodes(NODETYPE::HISTOGRAM);

  count_nodes_inner = statistics_count.getCountNodesInner();
  count_nodes_leaf = statistics_count.getCountNodesLeaf();

  depth_max = statistics_depth.getDepthMax();
  depth_min = statistics_depth.getDepthMin();
  depth_median = statistics_depth.getDepthMedian();
  depth_average = statistics_depth.getDepthAvg();

  SPDLOG_INFO("====================================");
  SPDLOG_INFO("|          SPN Statistics          |");
  SPDLOG_INFO("====================================");
  SPDLOG_INFO(" > Number of features: {}", count_features);
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

void GraphStatsCollection::initialize(ModuleOp&) {
  // Determine function name of the SPN (within the MLIR module)
  StringRef spnName = getSPNFuncNameFromModule();
  // Search the SPN root (as Operation) located in the function's definition.
  spn_root_global = getSPNRootByFuncName(spnName);

  // Collect statistics for node counts and their depth.
  statistics_count = GraphStatsNodeCount(spn_root_global);
  statistics_depth = GraphStatsNodeLevel(spn_root_global, 0);
}

StringRef GraphStatsCollection::getSPNFuncNameFromModule() {

  if (spn_func_name.empty()) {
    // Get all blocks of the provided module.
    for (auto& block : module->getBodyRegion().getBlocks()) {
      // Search block for FuncOps (i.e. function definitions).
      for (auto& op : block) {
        if (auto funcDef = dyn_cast<FuncOp>(op)) {
          for (auto& funcBlock : funcDef.getBody()) {
            // ToDo: Check 'getReverseIterator()' -- but it did not work properly with the LLVM build currently used.
            // Start search from the last operation of the function definition.
            for (auto it = funcBlock.rbegin(); it != funcBlock.rend(); --it) {
              // The FuncOp we are looking for is returned via a SPNSingleQueryOp.
              if (auto returnOp = dyn_cast<ReturnOp>(*it)) {
                // Check if a return-operand is a SPNSingleQueryOp; if yes -- return the corresponding StringRef.
                for (auto returnOperand : returnOp.getOperands()) {
                  if (auto query = dyn_cast<SPNSingleQueryOp>(returnOperand.getDefiningOp())) {
                    spn_func_name = query.spn();
                    return spn_func_name;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Return stored StringRef (possibly empty / dummy).
  return spn_func_name;
}

Operation* GraphStatsCollection::getSPNRootByFuncName(StringRef funcName) {
  if (funcName.empty()) {
    SPDLOG_WARN("Provided SPN function name was empty.");
  } else if (!spn_root_global) {
    // Get op of the SPN function.
    auto op = module->lookupSymbol(funcName);
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      FuncOp& f = funcOp;
      // The number of arguments is the actual feature count.
      count_features = f.getNumArguments();
      for (auto& funcBlock : f.getBody()) {
        // ToDo: Check 'getReverseIterator()' -- but it did not work properly with the LLVM build currently used.
        // Start search from the last operation of the function definition.
        for (auto it = funcBlock.rbegin(); it != funcBlock.rend(); --it) {
          // ToDo: Re-Check assumption 'Only one ReturnOp in SPNSingleQueryOp'.
          // The SPN-root we are looking for is the argument of the ReturnOp.
          if (auto returnOp = dyn_cast<ReturnOp>(*it)) {
            spn_root_global = returnOp.getOperand(0).getDefiningOp();
            return spn_root_global;
          }
        }
      }
    }
  }

  return spn_root_global;
}

StatsFile& GraphStatsCollection::execute() {
  if (!cached) {
    module = &input.execute();

    if (module != nullptr) {
      initialize(*module);
      collectGraphStats();
    } else {
      SPDLOG_WARN("Determined module is a nullptr, cannot write SPN statistics to file '{}'.", outfile.fileName());
    }

    cached = true;
  }

  return outfile;
}
