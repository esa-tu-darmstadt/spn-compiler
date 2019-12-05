//
// Created by lukas on 22.11.19.
//

#include "../include/runtime.h"

Kernel loadKernel(const std::string& fileName, const std::string& kernelName){
  return Kernel(fileName, kernelName);
}

void executeKernel(const Kernel& kernel, size_t num_elements, void* inputs, double* outputs){
  kernel.execute(num_elements, inputs, outputs);
}
