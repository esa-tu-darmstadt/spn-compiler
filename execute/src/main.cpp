//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <spnc.h>
#include <spnc-runtime.h>
#include <iostream>
#include <map>
#include "../../compiler/src/option/Options.h"

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
  // -1.44445
  // -2.65748
  // -0.670163
  // -1.70169
  // The expected results for categorical-example are
  // -2.43612
  // -1.51983
  // -1.98413
  // -1.06784
  // The expected results for gaussian-example are
  // -0.451583
  // -1.6469
  // -0.607833
  // -0.646895
  // Use the following input for mini-example:
  // int a[]{0, 0, 0, 1, 1, 0, 1, 1};
  // Use the following input for categorical-example:
  // char a[]{0, 0, 0, 1, 1, 0, 1, 1};
  // Use the following input for gaussian-example:
  //float a[]{0.5, 0.125, 0.125, 0.5, 0.25, 0.25, 0.325, 0.275};
  /*for(int i = 0; i < 4; ++i){
    double b[1];
    spnc_rt::spn_runtime::instance().execute(parseResult, 5, &a[i*2], b);
    std::cout << b[0] << std::endl;
  }*/
}