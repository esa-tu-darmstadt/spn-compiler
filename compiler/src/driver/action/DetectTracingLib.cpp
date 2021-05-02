//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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
