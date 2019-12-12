#include <unordered_set>
#include <graph-ir/GraphIRNode.h>
#include "llvm/IR/IRBuilder.h"
#include "transform/BaseVisitor.h"
#include <unordered_map>

using namespace llvm;

struct irVal {
  Value* val;
  int pos;
  size_t vec;
};

class IREmitter : public BaseVisitor {
public:
  IREmitter(
      std::unordered_map<std::string, size_t> &vec,
      std::unordered_map<size_t, std::unordered_set<size_t>>& directVecInputs,
      std::unordered_map<size_t, std::vector<NodeReference>>& vectors, Value *in,
      Function *func, LLVMContext &context, IRBuilder<> &builder,
      Module *module, unsigned width);

  void visitInputvar(InputVar &n, arg_t arg) override;

  void visitHistogram(Histogram &n, arg_t arg) override;

  void visitProduct(Product &n, arg_t arg) override;
  void visitSum(Sum &n, arg_t arg) override;
  void visitWeightedSum(WeightedSum &n, arg_t arg) override;

  std::unordered_map<std::string, irVal> getNodeMap();

private:
  std::unordered_map<std::string, size_t>& _vec;
  std::unordered_map<size_t, std::unordered_set<size_t>>& _directVecInputs;
  std::unordered_map<size_t, std::vector<NodeReference>>& _vectors;
  Value *_in;
  Function *_func;
  LLVMContext &_context;
  IRBuilder<> &_builder;
  Module *_module;
  std::unordered_map<std::string, irVal> node2value;
  std::unordered_map<std::string, Value *> input2value;
  unsigned simdWidth;
};
