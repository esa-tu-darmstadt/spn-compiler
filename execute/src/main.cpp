//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <spnc.h>
#include <spnc-runtime.h>
#include <iostream>
#include <map>

#ifndef TEST_KERNEL_DIR
#define TEST_KERNEL_DIR "/tmp"
#endif

int main(int argc, char* argv[]) {
  options_t options{{"target", "CPU"}, {"collect-graph-stats", "no"}, {"delete-temps", "false"}, {"bodyCodeGenMethod", "ILP"}, {"simdWidth", "4"}, {"iterativeSolving", "True"}};
  auto parseResult = spnc::spn_compiler::parseJSON(std::string(argv[1]), options);
  std::cout << "Parsed JSON? " << parseResult.fileName() << std::endl;
  Kernel kernel(std::string(TEST_KERNEL_DIR) + "/libdynamic-load-test.dylib", "foo");
  int a[]{1, 2, 3, 4, 5};
  double b[5];
  spnc_rt::spn_runtime::instance().execute(kernel, 5, a, b);
  for (auto d : b) {
    std::cout << d << std::endl;
  }
}
