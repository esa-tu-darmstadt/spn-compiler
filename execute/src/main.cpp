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
  // If the mini-example is fed to the compiler, the compiled kernel can be executed
  // using the following lines of code.
  // The expected results are 0.0625, 0.1875, 0.1875, 0.5625
  /*int a[]{0, 0, 0, 1, 1, 0, 1,1};
  for(int i = 0; i < 4; ++i){
    double b[1];
    spnc_rt::spn_runtime::instance().execute(parseResult, 5, &a[i*2], b);
    std::cout << b[0] << std::endl;
  }*/
}