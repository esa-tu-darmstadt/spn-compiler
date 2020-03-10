//
// Created by lukas on 20.11.19.
//

#ifndef SPNC_CODEGENBODY_H
#define SPNC_CODEGENBODY_H

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <graph-ir/GraphIRNode.h>
#include <graph-ir/IRGraph.h>

namespace spnc {
  using namespace llvm;

  typedef std::function<Value*(size_t inputIndex, Value* indVar)> InputVarValueMap;

  typedef std::function<Value*(Value* indVar)> OutputAddressMap;

  const static std::string TraceMDName = "spn.trace.nodeType";

  // Metadata-tags used to trace specific instructions. Values were set to simplify testing.
  enum class TraceMDTag : ushort { Sum = 0x1, WeightedSum = 0x2, Product = 0x4, Histogram = 0x8 };

  class CodeGenBody {

  public:
    CodeGenBody(Module& m, Function& f, IRBuilder<>& b) : module{m}, function{f}, builder{b} {}

    virtual Value* emitBody(IRGraph& graph, Value* indVar, InputVarValueMap inputs, OutputAddressMap output) = 0;

  protected:
    Module& module;
        Function& function;
        IRBuilder<>& builder;
    };
}


#endif //SPNC_CODEGENBODY_H
