//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <codegen/llvm-ir/transform/NumericalValueTracingPass.h>
#include "LLVMPipeline.h"
#include <llvm/Transforms/Utils/Cloning.h>
#include <iostream>
#include <driver/GlobalOptions.h>

using namespace spnc;

Option<bool> spnc::option::numericalTracing{"numerical-tracing", false,
                                            {depends(spnc::option::compilationTarget,
                                                     int(spnc::option::TargetMachine::CPU))}};

LLVMPipeline::LLVMPipeline(ActionWithOutput<Module>& _input, std::shared_ptr<LLVMContext> _llvmContext,
                           const Configuration& config) : ActionSingleInput<Module, Module>(_input),
                                                          llvmContext(std::move(_llvmContext)),
                                                          MPM{false}, MAM{false}, PB{} {
  if (option::numericalTracing.get(config)) {
    MPM.addPass(NumericalValueTracingPass());
  }
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
