//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <spnc-runtime.h>
#include <iostream>

int main(int argc, char* argv[]){
  auto parseResult = spnc::spn_compiler::parseJSON(std::string(argv[1]));
  std::cout << "Parsed JSON? " << parseResult.fileName() << std::endl;
  Kernel kernel("/home/wimi/ls/Code/SPN/spn-compiler-v2/cmake-build-debug/execute/libdynamic-load-test.so", "foo");
  Kernel kernel_mh("/home/mhalk/hiwi/test.so", "spn_kernel");
  int a[]{1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 1, 2, 3, 4, 5, 10, 20, 30, 40, 71, 160, 2, 3, 4, 159};
  double b[5];
  spnc_rt::spn_runtime::instance().execute(kernel_mh, 5, a, b);
  for(auto d : b){
    std::cout << d << std::endl;
  }
}