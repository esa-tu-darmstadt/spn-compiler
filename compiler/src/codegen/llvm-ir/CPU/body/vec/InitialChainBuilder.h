#include <graph-ir/GraphIRNode.h>
#include "graph-ir/transform/BaseVisitor.h"
#include <set>
#include <unordered_set>
#include <unordered_map>

using namespace spnc;

struct SIMDChain {
  size_t gatherCount;
  size_t length;
  std::vector<size_t> bottomLanes;
};
class InitialChainBuilder : public BaseVisitor {
 public:
  InitialChainBuilder(size_t width);
  void performInitialBuild(NodeReference root);
  void visitHistogram(Histogram &n, arg_t arg) override;
  void visitGauss(Gauss &n, arg_t arg) override;

  void visitProduct(Product &n, arg_t arg) override;

  void visitSum(Sum &n, arg_t arg) override;

  void visitWeightedSum(WeightedSum &n, arg_t arg) override;
  void
  findConflicts(std::vector<size_t> &chains,
                std::unordered_map<size_t, std::vector<size_t>> &conflicts);

  void
  findChainConflicts(
    std::vector<size_t> &chains,
    std::unordered_map<size_t, std::vector<size_t>> &nodeConflicts,
    std::unordered_map<size_t, std::set<size_t>> &chainConflicts);


  void generateCandidateChains( std::vector<size_t>& chains, std::unordered_map<size_t, std::set<size_t>>& conflicts, size_t chainsGoal);

  size_t recChainGen(std::vector<size_t> selected, std::vector<size_t> stillAvailable, std::unordered_map<size_t, std::set<size_t>>& conflicts);

  std::vector<std::vector<size_t>> scalarChains;
  
  std::vector<size_t> scalarSumChain;
  std::vector<size_t> scalarWeightedSumChain;
  std::vector<size_t> scalarProductChain;
  // TODO make all of these maps vectors
  std::unordered_map<size_t, std::vector<size_t>> sumConflicts;
  std::unordered_map<size_t, std::vector<size_t>> weightedSumConflicts;
  std::unordered_map<size_t, std::vector<size_t>> productConflicts;
  std::unordered_map<size_t, std::set<size_t>> sumChainConflicts;
  std::unordered_map<size_t, std::set<size_t>> weightedSumChainConflicts;
  std::unordered_map<size_t, std::set<size_t>> productChainConflicts;
  std::unordered_map<size_t, std::unordered_set<size_t>> dependsOn;
  std::vector<SIMDChain> candidateSIMDChains;
  std::unordered_map<size_t, size_t> childParentMap;
  std::unordered_map<size_t, std::vector<size_t>> parentChildrenMap;
  std::unordered_map<size_t, std::vector<size_t>> parentChildrenHistoMap;
  std::unordered_map<size_t, std::vector<size_t>> parentChildrenGaussMap;
  std::unordered_map<size_t, std::vector<size_t>> childChains;
  // getBestSet relies on nodes[0] being the root of the tree
  std::vector<NodeReference> nodes;
  std::vector<size_t> leafs;
  std::unordered_set<size_t> coveredPreLeafNodes;
  size_t width;
  size_t candidatesGoal;
};
