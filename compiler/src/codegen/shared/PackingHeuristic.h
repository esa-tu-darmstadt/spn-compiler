#include "codegen/shared/Packer.h"
#include "InitialChainBuilder.h"

#include <codegen/llvm-ir/CostInfo.h>

struct SIMDChainSet {
  std::vector<size_t> SIMDChains;
  int originatingChain;
  int posInOriginatingChain;
};

struct setSolverRes {
  std::vector<size_t> selectedSets;
  size_t cost;
  std::vector<vectorizationResultInfo> subtrees;
};

class PackingHeuristic : public Packer {
public:
  vectorizationResultInfo getVectorization(IRGraph &graph, size_t width);
 private:
  setSolverRes returnBestSet(
    std::vector<size_t> candidates, std::vector<SIMDChainSet> &simdChainSets,
    InitialChainBuilder &icb,
    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>>
    &chainPosToPackVecMap, std::vector<size_t> source, std::set<size_t> pruned);

  std::vector<std::pair<size_t, int>>
    orderByPotential(std::vector<SIMDChain> &chains, std::vector<size_t> toOrder, InitialChainBuilder& icb, bool first);

  std::vector<SIMDChainSet>
  buildSIMDChainSets(std::vector<size_t> &candidateSIMDChains, bool first,
                     InitialChainBuilder &icb, size_t setsGoal);

  std::pair<vectorizationResultInfo, size_t>
  getVectorizationRec(size_t rootNode, InitialChainBuilder &icb);
  std::unique_ptr<CostInfo> ci;
  std::unordered_map<size_t, std::pair<vectorizationResultInfo, size_t>> subTreeCache;
  size_t overallTreeSize;
};
