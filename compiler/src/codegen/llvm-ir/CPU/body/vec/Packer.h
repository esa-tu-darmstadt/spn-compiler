#pragma once
  
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <graph-ir/GraphIRNode.h>
#include <driver/GlobalOptions.h>
#include <graph-ir/IRGraph.h>

using namespace spnc;

struct vectorizationResultInfo {
  std::unordered_map<std::string, size_t> partOf;
  std::unordered_map<size_t, std::unordered_set<size_t>> directVecInputs;
  std::vector<std::vector<NodeReference>> vectors;
};


class Packer {
  virtual vectorizationResultInfo
  getVectorization(IRGraph &graph, size_t width, const Configuration& config) = 0;

};
