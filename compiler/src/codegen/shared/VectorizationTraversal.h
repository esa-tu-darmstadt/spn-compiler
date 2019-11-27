#include <graph-ir/GraphIRNode.h>
#include <stack>
#include "transform/BaseVisitor.h"
#include <unordered_set>

class VectorizationTraversal : public BaseVisitor {

public:
  VectorizationTraversal(std::unordered_set<std::string> pruned) : _pruned(pruned) {}
  std::vector<std::pair<std::string, std::vector<NodeReference>>>
  collectPaths(const NodeReference &rootNode);

  void visitHistogram(Histogram &n, arg_t arg) override;

  void visitProduct(Product &n, arg_t arg) override;

  void visitSum(Sum &n, arg_t arg) override;

  void visitWeightedSum(WeightedSum &n, arg_t arg) override;

private:
    std::vector<std::pair<std::string, std::vector<NodeReference>>> _paths;
    std::unordered_set<std::string> _pruned;
};
