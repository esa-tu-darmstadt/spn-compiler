//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

#include "pipeline/PipelineStep.h"
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <util/FileSystem.h>

namespace spnc {

///
/// Step to translate LLVM IR module into object code for the native CPU target.
/// The object code is written to the specified object file (*.o)
class EmitLLVMIR : public StepDualInput<EmitLLVMIR, llvm::Module, LLVMIR>,
                   public StepWithResult<LLVMIR> {

public:
  using StepDualInput<EmitLLVMIR, llvm::Module, LLVMIR>::StepDualInput;

  ExecutionResult executeStep(llvm::Module *module, LLVMIR *file);

  LLVMIR *result() override;

  STEP_NAME("emit-llvm-ir")

private:
  LLVMIR *outFile;
};

} // namespace spnc