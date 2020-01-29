#include "HeuristicChainBuilder.h"
#include "util/GraphIRTools.h"
#include <iostream>

HeuristicChainBuilder::HeuristicChainBuilder(IRGraph &graph, size_t width_)
    : width(width_) {
  graph.rootNode->accept(*this, {});
  std::cout << "test2" << std::endl;
  /*
  for (auto &chain : chains) {
    std::cout << "new chain" << std::endl;
    for (auto &op : chain.ops) {
      for (auto &lane : op) {
        std::cout << lane->id() << ", ";
      }
      std::cout << std::endl;
    }
    }*/
}

void HeuristicChainBuilder::buildRoots(std::vector<std::shared_ptr<chainNode>> incompleteFront, size_t width, std::vector<NodeReference>::iterator remainingCandidates, std::vector<NodeReference>::iterator vecEnd) {
  if (incompleteFront.size() == width) {
    addNewFront(incompleteFront, 0);
    return;
  }
  while(remainingCandidates != vecEnd) {
    auto newFront = incompleteFront;
    newFront.push_back(std::make_shared<chainNode>(chainNode{std::shared_ptr<chainNode>(), *remainingCandidates, 0}));
    buildRoots(newFront, width, ++remainingCandidates, vecEnd);
  }
}

void HeuristicChainBuilder::addNewFront(std::vector<std::shared_ptr<chainNode>>& front, size_t changedLane) {
  size_t id = fronts.size();
  for (int j = 0; j < front.size(); j++) {
    auto &lane = front[j];
    nodeToFronts[lane->node->id()].push_back({id, j});
  }
  fronts.push_back({front, changedLane});
}
void HeuristicChainBuilder::visitHistogram(Histogram &n, arg_t arg) {
}

void HeuristicChainBuilder::visitProduct(Product &n, arg_t arg) {
  visitNode(n);
}

void HeuristicChainBuilder::visitSum(Sum &n, arg_t arg) {
  visitNode(n);
}

void HeuristicChainBuilder::visitWeightedSum(WeightedSum &n, arg_t arg) {
  visitNode(n);
}

std::vector<SIMDOperationChain> HeuristicChainBuilder::getChains() {return chains;}
