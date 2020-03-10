//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <cstring>
#include <stdexcept>
#include "DetectTracingLib.h"

namespace spnc {

  DetectTracingLib::DetectTracingLib() : ActionWithOutput<BitcodeFile>() {}

  spnc::BitcodeFile& DetectTracingLib::execute() {
    if (!cached) {
      char* temp = getenv("SPNC_PATH_TRACE_LIB");
      std::string traceLibPath;

      if ((temp != nullptr) && (strlen(temp) > 0)) {
        traceLibPath = temp;
      } else {
        error = true;
      }

      outFile = std::make_unique<BitcodeFile>(traceLibPath, false);
      cached = true;
    }

    if (error) {
      throw std::runtime_error("Environmental variable 'SPNC_PATH_TRACE_LIB' not set or empty.");
    }

    return *outFile;
  }

}