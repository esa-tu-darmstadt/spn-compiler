//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <iostream>

int main(int argc, char* argv[]){
  auto parseResult = spnc::parseJSON(argc, argv);
  std::cout << "Parsed JSON? " << parseResult << std::endl;
}
