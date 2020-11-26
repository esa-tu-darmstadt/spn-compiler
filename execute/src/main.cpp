//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <spnc.h>
#include <spnc-runtime.h>
#include <iostream>
#include <map>
#include <driver/Options.h>

#ifndef TEST_KERNEL_DIR
#define TEST_KERNEL_DIR "/tmp"
#endif

int main(int argc, char* argv[]) {
  CLI::App app{"SPNC CLI"};
  spnc::interface::Options::registerCLOptions(app);
  CLI11_PARSE(app, argc, argv);

  auto options = spnc::interface::Options::collectCLOptions(app);
  auto parseResult = spnc::spn_compiler::compileQuery(std::string(argv[1]), options);
  std::cout << "Compiled kernel into file " << parseResult.fileName() << std::endl;

  //
  // Simple test to see if the compiled kernels are executable via the runtime.
  // If the mini-example or the categorical-example is fed to the compiler, the compiled kernel can be executed
  // using the following lines of code.
  // The expected results for mini-example are
  // 0.235875
  // 0.070125
  // 0.511625
  // 0.182375
  // The expected results for categorical-example are
  // 0.0875
  // 0.21875
  // 0.1375
  // 0.34375
  //
  // Use the following input for mini-example:
  // int a[]{0, 0, 0, 1, 1, 0, 1, 1};
  // Use the following input for categorical-example:
  //char a[]{0, 0, 0, 1, 1, 0, 1, 1};
  /*for(int i = 0; i < 4; ++i){
    double b[1];
    spnc_rt::spn_runtime::instance().execute(parseResult, 5, &a[i*2], b);
    std::cout << b[0] << std::endl;
  }*/
}