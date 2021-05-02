//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_LLVMLINKER_H
#define SPNC_LLVMLINKER_H

#include <util/FileSystem.h>
#include <driver/Actions.h>

namespace spnc {

  ///
  /// Action to invoke "llvm-link" to link two LLVM bitcode files/modules.
  class LLVMLinker : public ActionDualInput<LLVMBitcode, LLVMBitcode, LLVMBitcode> {

  public:
    /// Constructor.
    /// \param _input1 Action providing the first bitcode file.
    /// \param _input2 Action providing the secong bitcode file.
    /// \param outputFile File to write the output to.
    explicit LLVMLinker(ActionWithOutput<LLVMBitcode>& _input1,
                        ActionWithOutput<LLVMBitcode>& _input2,
                        LLVMBitcode outputFile);

    LLVMBitcode& execute() override;

  private:

    LLVMBitcode outFile;

    bool cached = false;

  };
}

#endif //SPNC_LLVMLINKER_H
