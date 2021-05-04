//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_CLANGKERNELLINKING_H
#define SPNC_CLANGKERNELLINKING_H

#include <driver/Actions.h>
#include <util/FileSystem.h>
#include <driver/Job.h>
#include <llvm/ADT/ArrayRef.h>
#include "../../../../common/include/Kernel.h"
#include "llvm/ADT/SmallVector.h"

namespace spnc {

  ///
  /// Action to turn an object (*.o) into a Kernel (shared object, *.so) using clang,
  // and running the linking to external libraries.
  class ClangKernelLinking : public ActionSingleInput<ObjectFile, Kernel> {

  public:

    /// Constructor.
    /// \param _input Action generating the object-file as input.
    /// \param outputFile File to write resulting kernel (shared object) to.
    /// \param kernelFunctionName Name of the top-level SPN function inside the object file.
    ClangKernelLinking(ActionWithOutput<ObjectFile>& _input, SharedObject outputFile,
                       std::shared_ptr<KernelInfo> info, llvm::ArrayRef<std::string> additionalLibraries = {},
                       llvm::ArrayRef<std::string> searchPaths = {});

    Kernel& execute() override;

  private:

    SharedObject outFile;

    std::shared_ptr<KernelInfo> kernelInfo;

    std::unique_ptr<Kernel> kernel;

    bool cached = false;

    llvm::SmallVector<std::string, 3> additionalLibs;

    llvm::SmallVector<std::string, 3> libSearchPaths;

  };

}

#endif //SPNC_CLANGKERNELLINKING_H
