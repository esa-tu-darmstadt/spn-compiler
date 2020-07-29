//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "GraphStats.h"

using namespace mlir;
using namespace mlir::spn;
using namespace spnc;

GraphStats::GraphStats(ActionWithOutput<ModuleOp>& _input, StatsFile outputFile)
    : ActionSingleInput<ModuleOp, StatsFile>{_input}, outfile{std::move(outputFile)} {
}

void GraphStats::collectGraphStats(ModuleOp& m) {
  std::vector<NODETYPE> nodetype_inner = {NODETYPE::SUM, NODETYPE::PRODUCT};
  std::vector<NODETYPE> nodetype_leaf = {NODETYPE::HISTOGRAM};

  spn_node_stats = {{NODETYPE::SUM, std::multiset<int>()},
                    {NODETYPE::PRODUCT, std::multiset<int>()},
                    {NODETYPE::HISTOGRAM, std::multiset<int>()}};

  std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({0}));

  StringRef spnName = getSPNFuncNameFromModule();
  auto root = getSPNRootByFuncName(spnName);
  visitNode(root, passed_arg);

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
    depth_min = *nodes.begin();
    depth_max = *(--nodes.end());

    // Since the used multimap is ordered, we can simply use the respective node count to get the median index.
    int median_index_temp = count_node_temp / 2;

    // ToDo: Determining the median depth has to be revisited once other leaf nodes are supported (2020-JAN-31).
    for (auto& level : nodes) {
      // level = node.first;
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

StringRef GraphStats::getSPNFuncNameFromModule() {

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

Operation* GraphStats::getSPNRootByFuncName(StringRef funcName) {
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

void GraphStats::visitNode(Operation* op, const arg_t& arg) {
  if (op == nullptr) {
    // Encountered nullptr -- abort.
    return;
  }

  bool hasOperands = false;
  int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;

  if (dyn_cast<SumOp>(op)) {
    hasOperands = true;
    ++count_nodes_sum;
    spn_node_stats.find(NODETYPE::SUM)->second.insert(currentLevel);
  } else if (dyn_cast<ProductOp>(op)) {
    hasOperands = true;
    ++count_nodes_product;
    spn_node_stats.find(NODETYPE::PRODUCT)->second.insert(currentLevel);
  } else if (dyn_cast<HistogramOp>(op)) {
    ++count_nodes_histogram;
    spn_node_stats.find(NODETYPE::HISTOGRAM)->second.insert(currentLevel);
  } else if (dyn_cast<ConstantOp>(op)) {
    // ToDo: Special handling of constants? Measure for improvement via optimizations?
  } else {
    // Encountered unhandled Op-Type
    std::cerr << "!UNHANDLED! -- OpDef: ";
    op->dump();
    std::cerr << std::endl;
    std::cerr << "!PARENT of UNHANDLED! -- OpDef: ";
    op->getParentOp()->dump();
    assert(false);
  }

  if (hasOperands) {
    for (auto child : op->getOperands()) {
      arg_t passed_arg(new GraphStatLevelInfo({currentLevel + 1}));
      visitNode(child.getDefiningOp(), passed_arg);
    }
  }

}

StatsFile& GraphStats::execute() {
  if (!cached) {
    module = &input.execute();

    if (module != nullptr) {
      collectGraphStats(*module);
    } else {
      SPDLOG_WARN("Produced module is a nullptr, cannot write stats to file '{}'.", outfile.fileName());
    }

    cached = true;
  }

  return outfile;
}
