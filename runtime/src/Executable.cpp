//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <dlfcn.h>
#include <iostream>
#include <util/Logging.h>
#include "Executable.h"
#include <omp.h>
#include <chrono>

using namespace spnc_rt;

Executable::Executable(const Kernel& _kernel) : kernel{&_kernel}, handle{nullptr}, kernel_func{nullptr} {}

Executable::Executable(spnc_rt::Executable&& other) noexcept : kernel{other.kernel}, handle{other.handle},
                                                               kernel_func{other.kernel_func} {
  other.handle = nullptr;
  other.kernel = nullptr;
}

Executable& Executable::operator=(Executable&& other) noexcept {
  kernel_func = other.kernel_func;
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

void Executable::execute(size_t num_elements, void* inputs, void* outputs) {
  if (!handle) {
    initialize();
  }
  if (kernel->target() == KernelTarget::CUDA) {
    executeGPU(num_elements, inputs, outputs);
  } else {
    assert(kernel->target() == KernelTarget::CPU);
    if (kernel->batchSize() == 1) {
      executeSingle(num_elements, inputs, outputs);
    } else {
      executeBatch(num_elements, inputs, outputs);
    }
  }

}

void Executable::executeSingle(size_t num_samples, void* inputs, void* outputs) {
  if (num_samples > 1) {
    SPDLOG_WARN("Executing a kernel optimized for single evaluation, computing only the first sample!");
  }
  assert(kernel_func);
  kernel_func(inputs, inputs, 0, 1, 1, kernel->numFeatures(), 1, outputs, outputs, 0, 1, 1);
}

void Executable::executeBatch(size_t num_samples, void* inputs, void* outputs) {
  assert(kernel_func);
  // Cast to char (i.e. byte) pointer to perform pointer arithmetic.
  char* input_ptr = reinterpret_cast<char*>(inputs);
  char* output_ptr = reinterpret_cast<char*>(outputs);
  size_t batchSize = kernel->batchSize();
  auto nativeStart = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for firstprivate(input_ptr, output_ptr, batchSize, num_samples)
  for (size_t i = 0; i < num_samples; i += batchSize) {
    // Calculate the number of remaining samples, can be < batchSize for the last batch.
    size_t remainingSamples = num_samples - i;
    size_t samples = std::min(batchSize, remainingSamples);
    //std::cout << "Remaining samples: " << remainingSamples << " now computing: " << samples << std::endl;
    // Calculate pointer to first input of this batch, using information about the
    // number of features and number of bytes used to encode the feature.
    char* input_offset = &(input_ptr[i * kernel->numFeatures() * kernel->bytesPerFeature()]);
    // Calculate pointer to first output for this batch, using information about the
    // number or results and number of bytes used to encode each result.
    char* output_offset = &(output_ptr[i * kernel->numResults() * kernel->bytesPerResult()]);
    kernel_func(input_offset, input_offset, 0, samples, 1, kernel->numFeatures(), 1,
                output_offset, output_offset, 0, samples, 1);
  }
  auto nativeEnd = std::chrono::high_resolution_clock::now();
  auto nativeTime = std::chrono::duration_cast<std::chrono::nanoseconds>(nativeEnd - nativeStart);
  std::cout << "SPEAKER_IDENT: NATIVE EXECUTION TIME: " << nativeTime.count() << " ns" << std::endl;
}

void Executable::executeGPU(size_t num_samples, void* inputs, void* outputs) {
  assert(kernel_func);
  // For GPUs, we launch all inputs at once. The host-part of the compiled kernel will split the samples
  // into multiple blocks, which in turn are processed by multiple GPU threads at once.
  kernel_func(inputs, inputs, 0, num_samples, 1, kernel->numFeatures(), 1, outputs, outputs, 0, num_samples, 1);
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
  *(void**) (&kernel_func) = dlsym(handle, kernel->kernelName().c_str());

  if ((error = dlerror()) != nullptr) {
    SPNC_FATAL_ERROR("Could not locate Kernel function {} in {}: {}", kernel->kernelName(), kernel->fileName(), error);
  }
}