//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
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

  ///
  /// Common interface for LLVM-IR based code-generation for the SPN loop body.
  class CodeGenBody {

  public:
    /// Constructor.
    /// \param m LLVM IR Module.
    /// \param f LLVM IR Function to insert the code into.
    /// \param b IRBuilder used to create and insert instructions.
    CodeGenBody(Module& m, Function& f, IRBuilder<>& b) : module{m}, function{f}, builder{b} {}

    /// Emit the given SPN graph as body into the function given to the constructor.
    /// \param graph SPN graph.
    /// \param indVar Loop induction variable of the surrounding loop.
    /// \param inputs Mapping of features (SPN input variables) to LLVM IR Values.
    /// \param output Mapping from loop index to output store address.
    /// \return Value of the loop induction variable after execution of the body.
    virtual Value* emitBody(IRGraph& graph, Value* indVar, InputVarValueMap inputs, OutputAddressMap output) = 0;

  protected:
    Module& module;
    Function& function;
    IRBuilder<>& builder;
  };
}

#endif //SPNC_CODEGENBODY_H
