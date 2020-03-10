//
// Created by ls on 2/12/20.
//

#include <codegen/llvm-ir/transform/NumericalValueTracingPass.h>
#include "LLVMPipeline.h"
#include <llvm/Transforms/Utils/Cloning.h>
#include <iostream>

namespace spnc {

  LLVMPipeline::LLVMPipeline(ActionWithOutput<Module>& _input, std::shared_ptr<LLVMContext> _llvmContext)
      : ActionSingleInput<Module, Module>(_input),
        llvmContext(std::move(_llvmContext)),
        MPM{false}, MAM{false}, PB{} {
    // TODO: Make addition of tracing pass dependent on configuration.
    MPM.addPass(NumericalValueTracingPass());
    PB.registerModuleAnalyses(MAM);
  }

  Module& LLVMPipeline::execute() {
    std::cout << "Executing LLVM Pipeline" << std::endl;
    if (!cached) {
      // Currently we explicitly clone the module to make sure that
      // other actions depending on the same input get to work with
      // the unmodified module.
      module = CloneModule(input.execute());
      MPM.run(*module, MAM);
    }
    return *module;
  }

}