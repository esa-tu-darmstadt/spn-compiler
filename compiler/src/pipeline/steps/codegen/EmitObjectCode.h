//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_DRIVER_ACTION_EMITOBJECTCODE_H
#define SPNC_COMPILER_SRC_DRIVER_ACTION_EMITOBJECTCODE_H

#include <util/FileSystem.h>
#include "pipeline/PipelineStep.h"
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

namespace spnc {

  ///
  /// Action to translate LLVM IR module into object code for the native CPU target.
  /// The object code is written to the specified object file (*.o)
  class EmitObjectCode : public StepDualInput<EmitObjectCode, llvm::Module, ObjectFile>,
                         public StepWithResult<ObjectFile> {

  public:

    using StepDualInput<EmitObjectCode, llvm::Module, ObjectFile>::StepDualInput;

    ExecutionResult executeStep(llvm::Module* module, ObjectFile* file);

    ObjectFile* result() override;

    STEP_NAME("emit-object-code")

  private:

    ObjectFile* outFile;

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_ACTION_EMITOBJECTCODE_H
