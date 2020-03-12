//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_CODEGENLOOP_H
#define SPNC_CODEGENLOOP_H

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <graph-ir/GraphIRNode.h>

using namespace llvm;

namespace spnc {

  ///
  /// Common interface for LLVM-IR based code-generation for the loop iterating over the SPN queries
  /// inside a batch.
  class CodeGenLoop {
  public:

    /// Constructor.
    /// \param m LLVM IR Module.
    /// \param g SPN graph.
    CodeGenLoop(Module& m, IRGraph& g) : module{m}, graph{g} {}

    /// Trigger the code-generation
    /// \param function LLVM IR function to insert the loop into.
    /// \param builder IRBuilder to create IR instructions.
    /// \param lowerBound LLVM IR Value corresponding to the inclusive lower bound of the loop.
    /// \param upperBound LLVM IR Value corresponding to the exclusive upper bound of the loop.
    virtual void emitLoop(Function& function, IRBuilder<>& builder, Value* lowerBound, Value* upperBound) = 0;

    /// Construct the input arguments for the surrounding function.
    /// \return List of types to insert into the signature of the surrounding function for the inputs.
    virtual std::vector<Type*> constructInputArgumentTypes() = 0;

    /// Construct the output arguments for the surrounding function.
    /// \return List of types to insert into the signature of the surrounding function for the outputs.
    virtual std::vector<Type*> constructOutputArgumentTypes() = 0;

  protected:

    ///
    /// Surrounding LLVM IR Module.
    Module& module;

    ///
    /// SPN graph.
    IRGraph& graph;

  };
}

#endif //SPNC_CODEGENLOOP_H
