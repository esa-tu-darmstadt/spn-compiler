//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <iostream>

int main(int argc, char* argv[]){
    auto parseResult = spnc::spnc::parseJSON(std::string(argv[1]));
    std::cout << "Parsed JSON? " << parseResult << std::endl;
}