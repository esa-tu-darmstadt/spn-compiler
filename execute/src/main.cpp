//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <spnc.h>
#include <spnc-runtime.h>
#include <iostream>
#include <map>

int main(int argc, char* argv[]) {
  options_t options{{"target", "CPU"}, {"collect-graph-stats", "no"}};
  auto parseResult = spnc::spn_compiler::parseJSON(std::string(argv[1]), options);
  std::cout << "Parsed JSON? " << parseResult.fileName() << std::endl;
  Kernel kernel("/home/wimi/ls/Code/SPN/spn-compiler-v2/cmake-build-debug/execute/libdynamic-load-test.so", "foo");
  int a[]{1, 2, 3, 4, 5};
  double b[5];
  spnc_rt::spn_runtime::instance().execute(kernel, 5, a, b);
  for (auto d : b) {
    std::cout << d << std::endl;
  }
}