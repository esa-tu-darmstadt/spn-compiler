//
// Created by ls on 12/5/19.
//
#include <iostream>

extern "C" {
  void foo(size_t num_elements, void *inputs, double *outputs) {
    std::cout << "Dynamically loaded function foo" << std::endl;
    int *int_inputs = (int *) inputs;
    for (int i = 0; i < num_elements; ++i) {
      outputs[i] = int_inputs[i] * 2.0;
    }
  }
}