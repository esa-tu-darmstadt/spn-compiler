//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_LLVMSTATICCOMPILER_H
#define SPNC_LLVMSTATICCOMPILER_H

#include <util/FileSystem.h>
#include <driver/Actions.h>

namespace spnc {

  ///
  /// Action to use the LLVM static compiler ("llc") to turn a LLVM bitcode file
  /// into an object file (*.o).
  class LLVMStaticCompiler : public ActionSingleInput<LLVMBitcode, ObjectFile> {

  public:

    /// Constructor.
    /// \param _input Action providing the input bitcode file.
    /// \param outputFile File to write output to.
    explicit LLVMStaticCompiler(ActionWithOutput<LLVMBitcode>& _input, ObjectFile outputFile);

    ObjectFile& execute() override;

  private:

    ObjectFile outFile;

    bool cached = false;

  };
}

#endif //SPNC_LLVMSTATICCOMPILER_H
