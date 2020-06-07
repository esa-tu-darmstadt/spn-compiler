#include "graph-ir/transform/BaseVisitor.h"
#include <graph-ir/GraphIRNode.h>
#include <queue>

using namespace spnc;

class BFSOrderProducer : public BaseVisitor {
public:
  void visitInputvar(InputVar &n, arg_t arg) override;
  void visitHistogram(Histogram &n, arg_t arg) override;
  void visitGauss(Gauss &n, arg_t arg) override;
  void visitProduct(Product &n, arg_t arg) override;
  void visitSum(Sum &n, arg_t arg) override;
  void visitWeightedSum(WeightedSum &n, arg_t arg) override;
  std::queue<std::pair<size_t, NodeReference>> q;
  size_t currentLevel = 0;
};
