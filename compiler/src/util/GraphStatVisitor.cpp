//
// Created by mh on 2020-JAN-27.
//

#include <fstream>
#include "GraphStatVisitor.h"

namespace spnc {

  GraphStatVisitor::GraphStatVisitor(ActionWithOutput<IRGraph>& _input, const std::string &outputFile)
      : ActionSingleInput<IRGraph, File<FileType::SPN_JSON>>{_input}, outfile{outputFile} {}

    void GraphStatVisitor::collectGraphStats(const NodeReference& rootNode) {
      std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({1, {}}));
      rootNode->accept(*this, passed_arg);

      count_nodes_inner = count_nodes_sum + count_nodes_product;
      count_nodes_leaf = count_nodes_histogram;

      std::cout << "====================================" << std::endl;
      std::cout << "|          SPN Statistics          |" << std::endl;
      std::cout << "====================================" << std::endl;
      std::cout << " > Number of features: " << count_features << std::endl;
      std::cout << " > Maximum depth: " << max_depth << std::endl;
      std::cout << " > Nodes (inner, leaf): (" << count_nodes_inner << ", " << count_nodes_leaf << ")" << std::endl;
      std::cout << " > Sum-Nodes: " << count_nodes_sum << std::endl;
      std::cout << " > Product-Nodes: " << count_nodes_product << std::endl;
      std::cout << " > Histogram-Nodes: " << count_nodes_histogram << std::endl;
      std::cout << "====================================" << std::endl;
      
      json stats;

      stats["count_features"] = count_features;
      stats["max_depth"] = max_depth;
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

    void GraphStatVisitor::visitInputvar(InputVar &n, arg_t arg) {}

    void GraphStatVisitor::visitHistogram(Histogram &n, arg_t arg) {
      ++count_nodes_histogram;
      int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;

      if (max_depth < currentLevel) {
        max_depth = currentLevel;
      }

      n.indexVar()->accept(*this, nullptr);
    }

    void GraphStatVisitor::visitProduct(Product &n, arg_t arg) {
      ++count_nodes_product;
      int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;

      for(auto& child : *n.multiplicands()){
        std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({currentLevel + 1, {}}));
        child->accept(*this, passed_arg);
      }
    }

    void GraphStatVisitor::visitSum(Sum &n, arg_t arg) {
      for(auto& child : *n.addends()){
        child->accept(*this, nullptr);
      }
    }

    void GraphStatVisitor::visitWeightedSum(WeightedSum &n, arg_t arg) {
      ++count_nodes_sum;
      int currentLevel = std::static_pointer_cast<GraphStatLevelInfo>(arg)->level;

      for(auto& child : *n.addends()){
        std::shared_ptr<void> passed_arg(new GraphStatLevelInfo({currentLevel + 1, {}}));
        child.addend->accept(*this, passed_arg);
      }
    }

    spnc::File<FileType::SPN_JSON>& spnc::GraphStatVisitor::execute() {
      if(!cached){
        IRGraph graph = input.execute();
        count_features = graph.inputs->size();
        collectGraphStats(graph.rootNode);
        cached = true;
      }
      return outfile;
    }

}
