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
/// Step to translate C++ LLVM IR source into object code for the IPU target.
/// The object code is written to the specified graph program (*.gp)
/// Accepts either FileType:CPP or FileType:LLVM_IR as its template parameter.
template <FileType SourceType>
class EmitObjectCodeForIPU
    : public StepDualInput<EmitObjectCodeForIPU<SourceType>, File<SourceType>, CompiledGraphProgram>,
      public StepWithResult<CompiledGraphProgram> {
  static_assert(SourceType == FileType::CPP || SourceType == FileType::LLVM_IR,
                "SourceType must be either CPP or LLVM_IR");

public:
  using StepDualInput<EmitObjectCodeForIPU, File<SourceType>, CompiledGraphProgram>::StepDualInput;

  ExecutionResult executeStep(File<SourceType> *source, CompiledGraphProgram *file);

  CompiledGraphProgram *result() override { return outFile; }

  STEP_NAME("emit-object-code-ipu")

private:
  CompiledGraphProgram *outFile;
};

} // namespace spnc