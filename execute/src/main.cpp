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
    double a[]{1., 2., 3., 4., 5.};
    double b[5];
    spnc_rt::spn_runtime::instance().execute(kernel, 5, a, b);
    for(auto d : b){
      std::cout << d << std::endl;
    }
}