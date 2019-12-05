//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <runtime.h>
#include <iostream>

int main(int argc, char* argv[]){
    auto parseResult = spnc::parseJSON(std::string(argv[1]));
    std::cout << "Parsed JSON? " << parseResult << std::endl;
    auto kernel = loadKernel("/home/wimi/ls/Code/SPN/spn-compiler-v2/cmake-build-debug/execute/libdynamic-load-test.so", "foo");
    int a[]{1, 2, 3, 4, 5};
    double b[5];
    executeKernel(kernel, 5, a, b);
    for(auto d : b){
      std::cout << d << std::endl;
    }
}