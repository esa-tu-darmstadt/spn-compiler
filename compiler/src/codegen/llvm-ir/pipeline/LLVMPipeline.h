//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_LLVMPIPELINE_H
#define SPNC_LLVMPIPELINE_H

#include <llvm/IR/Module.h>
#include <driver/Actions.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <driver/Options.h>

using namespace llvm;
using namespace spnc::interface;

namespace spnc {

  namespace option {
    ///
    /// Flag indicating whether instrumentation should be added to
    /// trace numerical values inside the SPN graph.
    extern Option<bool> numericalTracing;
  }

  ///
  /// Action to run a series of LLVM IR Passes on a LLVM IR Module.
  /// Creates a copy of the Module and uses the PassManager API to run
  /// a series of passes on the copied module.
  class LLVMPipeline : public ActionSingleInput<Module, Module> {

  public:

    /// Constructor.
    /// \param _input Input LLVM IR Module.
    /// \param _llvmContext LLVM IR context.
    /// \param config Compiler interface configuration.
    explicit LLVMPipeline(ActionWithOutput<Module>& _input, std::shared_ptr<LLVMContext> _llvmContext,
                          const Configuration& config);

    Module& execute() override;

  private:

    std::shared_ptr<LLVMContext> llvmContext;

    ModulePassManager MPM;

    ModuleAnalysisManager MAM;

    PassBuilder PB;

    std::unique_ptr<Module> module;

    bool cached = false;
  };
}

#endif //SPNC_LLVMPIPELINE_H
