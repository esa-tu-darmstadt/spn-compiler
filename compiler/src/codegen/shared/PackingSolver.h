#include <unordered_set>
#include <graph-ir/GraphIRNode.h>
#include <unordered_map>

class PackingSolver {
public:
  std::unordered_map<std::string, size_t> getPacking(IRGraph& graph, size_t width);
  std::unordered_map<size_t, std::unordered_set<size_t>> directVecInputs;
  std::unordered_map<size_t,std::vector<NodeReference>> vectors;
};
