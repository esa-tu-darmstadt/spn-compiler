//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_TRACINGLIB_H
#define SPNC_TRACINGLIB_H

#include <memory>
#include <util/FileSystem.h>
#include <driver/Actions.h>

namespace spnc {

  ///
  /// Action to locate the tracing library provided as LLVM bitcode library.
  class DetectTracingLib : public ActionWithOutput<LLVMBitcode> {

  public:

    LLVMBitcode& execute() override;

  private:

    std::unique_ptr<LLVMBitcode> outFile;

    bool cached = false;

    bool error = false;

  };
}

#endif //SPNC_TRACINGLIB_H
