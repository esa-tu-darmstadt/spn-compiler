//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

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
