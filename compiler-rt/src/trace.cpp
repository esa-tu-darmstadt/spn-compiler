//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <cstdlib>
#include <cstring>
#include <fstream>

bool initDone = false;
std::string pathTraceOut;
std::fstream traceOut;

void doInit() {
  char* temp = getenv("SPNC_PATH_TRACE_OUT");

  if ((temp == nullptr) || (strlen(temp) <= 0)) {
    // Use fallback value
    pathTraceOut = "/tmp/spn_trace_out.txt";
  } else {
    pathTraceOut = temp;
  }

  initDone = true;
}

extern "C" {
void trace(double d) {
  if (!initDone) {
    doInit();
  }

  traceOut.open(pathTraceOut, std::fstream::out | std::fstream::app);
  if (traceOut.is_open()) {
    traceOut << d << std::endl;
    traceOut.close();
  }
}
}
