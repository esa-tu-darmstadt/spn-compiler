#include "transform/BaseVisitor.h"
#include <graph-ir/GraphIRNode.h>
#include <unordered_map>
#include "gurobi_c++.h"
#include <queue>
#include <unordered_set>
#include <set>

std::vector<size_t>
flattenPack(std::vector<size_t> base, size_t limit,
            std::unordered_map<size_t, std::vector<size_t>> &fixedPacks);

struct vecVar {
  std::vector<size_t> lanes;
  GRBVar var;
};

class SchedulingConflictTraversal : public BaseVisitor {
public:
  SchedulingConflictTraversal(size_t width, GRBModel* m)
      : simdWidth(width), model(m) {};
  void visitInputvar(InputVar &n, arg_t arg) override;
  void visitHistogram(Histogram &n, arg_t arg) override;
  void visitProduct(Product &n, arg_t arg) override;
  void visitSum(Sum &n, arg_t arg) override;
  void visitWeightedSum(WeightedSum &n, arg_t arg) override;
  void findConflicts(NodeReference root);
  void setupNewIteration(std::unordered_set<size_t> prevVecs,
                         std::vector<size_t> preNonVecs, GRBModel* newModel);
  std::vector<size_t> getNewOpSet(std::vector<size_t> &old,
                     std::unordered_map<size_t, size_t> &componentToPackId);
  std::vector<size_t> updateConflicts(GraphIRNode &n, arg_t arg);
  void generateVariables();
  void genVarsRoot(std::vector<size_t> availableOps);
  void genVarsRec(std::vector<size_t> pack, std::vector<size_t> availableOps);
  // first: users, second: inputs
  std::vector<std::pair<std::set<size_t>, std::set<size_t>>> conflicts;
  std::unordered_map<std::string, size_t> idMap;
  std::vector<size_t> sums;
  std::vector<size_t> weightedSums;
  std::vector<size_t> products;
  std::vector<std::string> names;
  std::vector<vecVar> vecVars;
  std::unordered_map<size_t, std::vector<size_t>> partOf;
  std::unordered_map<size_t, GRBVar> serVars;
  std::unordered_map<size_t, std::vector<size_t>> fixedPacks;
  std::unordered_map<size_t, size_t> singleOpToFixedVec;
  // maps index into vecVars to index into conflicts/fixedPacks
  std::unordered_map<size_t, size_t> oldVecs;
  size_t simdWidth;
  GRBModel* model;
};

