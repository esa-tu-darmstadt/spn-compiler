//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_LLVMCPUCODEGEN_H
#define SPNC_LLVMCPUCODEGEN_H

#include <unordered_map>
#include <graph-ir/GraphIRNode.h>
#include <driver/Actions.h>
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace spnc {

  ///
  /// Action to invoke the LLVM IR code-generation. Turns an SPN-graph into a
  /// function looping over all queries in a batch and computing the SPN bottom-up.
  class LLVMCPUCodegen : public ActionSingleInput<IRGraph, llvm::Module> {

  public:
    /// Constructor.
    /// \param _input Input action providing the SPN graph.
    /// \param _kernelName Name of the top-level function to create.
    /// \param _llvmContext LLVMContext.
    LLVMCPUCodegen(ActionWithOutput<IRGraph>& _input, std::string _kernelName,
                   std::shared_ptr<LLVMContext> _llvmContext);

    llvm::Module& execute() override;

  private:

    void generateLLVMIR(IRGraph& graph);

    std::shared_ptr<LLVMContext> context;

    IRBuilder<> builder;

    std::unique_ptr<Module> module;

    std::string kernelName;

    std::unordered_map<std::string, Value*> node2value;

    bool cached = false;

  };
}

#endif //SPNC_LLVMCPUCODEGEN_H
