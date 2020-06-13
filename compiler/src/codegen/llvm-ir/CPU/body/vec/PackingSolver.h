#include "CostInfo.h"
#include <unordered_set>
#include "Packer.h"
#include "SchedulingConflictTraversal.h"
#include <unordered_map>
#include <set>
#include <driver/GlobalOptions.h>

class IndexRefMapper;

struct solverResult {
  std::unordered_set<size_t> vecs;
  std::vector<size_t> nonVecs;
  std::unordered_map<size_t, std::unordered_set<size_t>> directVecInputs;
};

class PackingSolver : public Packer {
public:
  vectorizationResultInfo getVectorization(IRGraph &graph, size_t width, const Configuration& config);
  // Of vector _first_, the names in _second_ are needed in order by _first_'s consumer
  std::unordered_map<size_t,std::unordered_set<size_t>> directVecInputs;
  std::vector<std::vector<NodeReference>> vectors;
 private:
   solverResult runSolver(
       std::vector<std::pair<std::set<size_t>, std::set<size_t>>> &conflicts,
       std::unordered_map<std::string, size_t> &idMap,
       std::vector<vecVar> &vecVars,
       std::unordered_map<size_t, std::vector<size_t>> &partOf,
       std::unordered_map<size_t, GRBVar> &serVars,
       std::unordered_map<size_t, std::vector<size_t>> &fixedPacks,
       std::unordered_set<std::string>& histograms, std::unordered_set<std::string>& gaussians,
       IndexRefMapper &irm,
       std::unordered_map<size_t, size_t> &singleOpToFixedVec, GRBModel& model);
  std::unique_ptr<CostInfo> ci;
};
