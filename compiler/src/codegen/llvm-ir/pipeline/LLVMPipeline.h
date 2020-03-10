//
// Created by ls on 2/12/20.
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
    extern Option<bool> numericalTracing;
  }

  class LLVMPipeline : public ActionSingleInput<Module, Module> {

  public:

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
