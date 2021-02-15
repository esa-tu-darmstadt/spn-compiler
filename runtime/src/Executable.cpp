//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <dlfcn.h>
#include <iostream>
#include <util/Logging.h>
#include "Executable.h"

using namespace spnc_rt;

Executable::Executable(const Kernel& _kernel) : kernel{&_kernel}, handle{nullptr}, kernel_func{nullptr} {}

Executable::Executable(spnc_rt::Executable&& other) noexcept : kernel{other.kernel}, handle{other.handle},
                                                               kernel_func{other.kernel_func} {
  other.kernel_func.single = nullptr;
  other.handle = nullptr;
  other.kernel = nullptr;
}

Executable& Executable::operator=(Executable&& other) noexcept {
  kernel_func = other.kernel_func;
  other.kernel_func.single = nullptr;
  handle = other.handle;
  other.handle = nullptr;
  kernel = other.kernel;
  other.kernel = nullptr;
  return *this;
}

Executable::~Executable() {
  if (handle) {
    dlclose(handle);
  }
}

void Executable::execute(size_t num_elements, void* inputs, double* outputs) {
  if (!handle) {
    initialize();
  }
  if (kernel->batchSize() == 1) {
    executeSingle(num_elements, inputs, outputs);
  } else {
    executeBatch(num_elements, inputs, outputs);
  }

}

void Executable::executeSingle(size_t num_samples, void* inputs, double* outputs) {
  if (num_samples > 1) {
    SPDLOG_WARN("Executing a kernel optimized for single evaluation, computing only the first sample!");
  }
  assert(kernel_func.single);
  kernel_func.single(inputs, inputs, 0, kernel->numFeatures(), 1, outputs, outputs, 0, 1, 1);
}

void Executable::executeBatch(size_t num_samples, void* inputs, double* outputs) {
  assert(kernel_func.batch);
  // Cast to char (i.e. byte) pointer to perform pointer arithmetic.
  char* input_ptr = reinterpret_cast<char*>(inputs);
  size_t batchSize = kernel->batchSize();
  for (size_t i = 0; i < num_samples; i += batchSize) {
    // Calculate the number of remaining samples, can be < batchSize for the last batch.
    size_t remainingSamples = num_samples - i;
    size_t samples = std::min(batchSize, remainingSamples);
    std::cout << "Remaining samples: " << remainingSamples << " now computing: " << samples << std::endl;
    // Calculate pointer to first input of this batch, using information about the
    // number of features and number of bytes used to encode the feature.
    char* offset_ptr = &(input_ptr[i * kernel->numFeatures() * kernel->bytesPerFeature()]);
    // Calculate pointer to first output for this batch.
    double* output_ptr = &outputs[i];
    kernel_func.batch(samples, offset_ptr, offset_ptr, 0, samples, 1, kernel->numFeatures(), 1,
                      output_ptr, output_ptr, 0, samples, 1);
  }
}

void Executable::initialize() {
  char* error;
  // Try to open the shared object file.
  handle = dlopen(kernel->fileName().c_str(), RTLD_LAZY);
  if (!handle) {
    SPN_RT_FATAL_ERROR("Error opening Kernel file {}: {}", kernel->fileName(), dlerror());
  }

  // Clear existing errors.
  dlerror();

  // Try to locate a function with the given name in the shared object.
  if (kernel->batchSize() == 1) {
    *(void**) (&kernel_func.single) = dlsym(handle, kernel->kernelName().c_str());
  } else {
    *(void**) (&kernel_func.batch) = dlsym(handle, kernel->kernelName().c_str());
  }

  if ((error = dlerror()) != nullptr) {
    SPNC_FATAL_ERROR("Could not locate Kernel function {} in {}: {}", kernel->kernelName(), kernel->fileName(), error);
  }
}