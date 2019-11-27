//
// Created by ls on 10/9/19.
//

#ifndef SPNC_LLVMCODEGEN_H
#define SPNC_LLVMCODEGEN_H

#include <unordered_map>
#include <graph-ir/GraphIRNode.h>
#include "llvm/IR/IRBuilder.h"
#include <unordered_set>

using namespace llvm;

class LLVMCodegen {

public:
  explicit LLVMCodegen();

  void generateLLVMIR(IRGraph &graph, bool vectorize);

private:
  void emitBody(IRGraph &graph, Value *in, Value *out);
  void emitVecBody(IRGraph &graph, Value *in, Value *out);
  void emitInput(InputVar* n, Value* addr);
  void emitHistogram(Histogram* n);
  void emitProduct(Product* n);
  void emitSum(Sum* n);
  void emitWeightedSum(WeightedSum* n);
  void emitStore(GraphIRNode* n, Value* addr);
  std::vector<std::vector<NodeReference>>
  getLongestChain(std::vector<NodeReference> roots,
                  std::unordered_set<std::string> pruned);
  LLVMContext context;
  IRBuilder<> builder;
  std::unique_ptr<Module> module;
  std::unordered_map<std::string, Value *> node2value;
  std::unordered_map<std::string, Value*> input2value;
  Function* func;
};


#endif //SPNC_LLVMCODEGEN_H
