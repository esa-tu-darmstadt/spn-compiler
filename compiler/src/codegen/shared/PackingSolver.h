#include <unordered_set>
#include "codegen/shared/Packer.h"
#include "codegen/shared/SchedulingConflictTraversal.h"
#include <unordered_map>
#include <set>

class IndexRefMapper;

struct solverResult {
  std::unordered_set<size_t> vecs;
  std::vector<size_t> nonVecs;
  std::unordered_map<size_t, std::unordered_set<size_t>> directVecInputs;
};

class PackingSolver : public Packer {
public:
  vectorizationResultInfo getVectorization(IRGraph &graph, size_t width);
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
       IndexRefMapper &irm,
       std::unordered_map<size_t, size_t> &singleOpToFixedVec, GRBModel& model);
};
