//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_CLANGKERNELLINKING_H
#define SPNC_CLANGKERNELLINKING_H

#include "Kernel.h"
#include "pipeline/PipelineStep.h"
#include "util/FileSystem.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace spnc {

///
/// Step to turn an object (*.o) into a Kernel (shared object, *.so) using
/// clang,
// and running the linking to external libraries.
class ClangKernelLinking
    : public StepDualInput<ClangKernelLinking, ObjectFile, SharedObject>,
      public StepWithResult<Kernel> {

public:
  using StepDualInput<ClangKernelLinking, ObjectFile,
                      SharedObject>::StepDualInput;

  ExecutionResult executeStep(ObjectFile *objectFile,
                              SharedObject *sharedObject);

  Kernel *result() override;

  STEP_NAME("kernel-linking")

private:
  SharedObject *outFile = nullptr;

  std::unique_ptr<Kernel> kernel;
};

} // namespace spnc

#endif // SPNC_CLANGKERNELLINKING_H
