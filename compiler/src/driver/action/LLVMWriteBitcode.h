//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_LLVMWRITEBITCODE_H
#define SPNC_LLVMWRITEBITCODE_H

#include <llvm/IR/Module.h>
#include <driver/BaseActions.h>

namespace spnc {

  ///
  /// Action to write an LLVM IR Module into a bitcode file.
  class LLVMWriteBitcode : public ActionSingleInput<llvm::Module, LLVMBitcode> {

  public:

    /// Constructor.
    /// \param _input Action providing the input LLVM IR module.
    /// \param outputFile File to write output to.
    explicit LLVMWriteBitcode(ActionWithOutput<llvm::Module>& _input, LLVMBitcode outputFile);

    LLVMBitcode& execute() override;

  private:

    LLVMBitcode outFile;

    bool cached = false;

  };
}

#endif //SPNC_LLVMWRITEBITCODE_H
