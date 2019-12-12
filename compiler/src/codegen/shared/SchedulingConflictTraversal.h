#include "transform/BaseVisitor.h"
#include <graph-ir/GraphIRNode.h>
#include <unordered_map>
#include "gurobi_c++.h"


struct vecVar {
  std::vector<size_t> lanes;
  GRBVar var;
};

class SchedulingConflictTraversal : public BaseVisitor {
public:
  SchedulingConflictTraversal(size_t width, GRBModel& m)
      : simdWidth(width), model(m) {};
  void visitInputvar(InputVar &n, arg_t arg) override;
  void visitHistogram(Histogram &n, arg_t arg) override;
  void visitProduct(Product &n, arg_t arg) override;
  void visitSum(Sum &n, arg_t arg) override;
  void visitWeightedSum(WeightedSum &n, arg_t arg) override;
  void findConflicts(NodeReference root);
  std::vector<size_t> updateConflicts(GraphIRNode &n, arg_t arg);
  void generateVariables();
  void genVarsRoot(std::vector<size_t> availableOps);
  void genVarsRec(std::vector<size_t> pack, std::vector<size_t> availableOps);
  // first: users, second: inputs
  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> conflicts;
  std::unordered_map<std::string, size_t> idMap;
  std::vector<size_t> sums;
  std::vector<size_t> weightedSums;
  std::vector<size_t> products;
  std::vector<std::string> names;
  std::vector<vecVar> vecVars;
  std::vector<std::vector<size_t>> partOf;
  std::vector<GRBVar> serVars;
  size_t simdWidth;
  GRBModel& model;
};

