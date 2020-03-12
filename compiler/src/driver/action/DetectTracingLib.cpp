//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <cstring>
#include <stdexcept>
#include <util/Logging.h>
#include "DetectTracingLib.h"

using namespace spnc;

LLVMBitcode& DetectTracingLib::execute() {
  if (!cached) {
    // Try to load the location of the bitcode library from the environment variable.
    char* temp = getenv("SPNC_PATH_TRACE_LIB");
    std::string traceLibPath;

    if ((temp != nullptr) && (strlen(temp) > 0)) {
      traceLibPath = temp;
    } else {
      error = true;
    }

    outFile = std::make_unique<LLVMBitcode>(traceLibPath, false);
    cached = true;
  }

  if (error) {
    SPNC_FATAL_ERROR("\"Environmental variable 'SPNC_PATH_TRACE_LIB' not set or empty.");
  }

  return *outFile;
}
