//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <fstream>

extern "C" {
void trace(double d) {
  char filename[] = "/tmp/trace.txt";
  std::fstream file(filename, std::fstream::in | std::fstream::out | std::fstream::app);
  if (file.is_open()) {
    file << d << std::endl;
    file.close();
  }
}
}
