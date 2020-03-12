//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_CLANGKERNELLINKING_H
#define SPNC_CLANGKERNELLINKING_H

#include <driver/Actions.h>
#include <util/FileSystem.h>
#include "../../../../common/include/Kernel.h"

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
                       const std::string& kernelFunctionName);

    Kernel& execute() override;

  private:

    SharedObject outFile;

    std::string kernelName;

    Kernel kernel;

    bool cached = false;

  };

}

#endif //SPNC_CLANGKERNELLINKING_H
