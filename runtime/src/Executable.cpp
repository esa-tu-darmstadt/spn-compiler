//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <dlfcn.h>
#include <iostream>
#include <util/Logging.h>
#include "Executable.h"
#include <omp.h>
#include <chrono>

using namespace spnc_rt;

Executable::Executable(const Kernel& _kernel) : kernel{&_kernel}, handle{nullptr}, kernel_func{nullptr} {}

Executable::Executable(spnc_rt::Executable&& other) noexcept: kernel{other.kernel}, handle{other.handle},
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
  } else if (kernel->target() == KernelTarget::CPU) {
    if (kernel->batchSize() == 1) {
      executeSingle(num_elements, inputs, outputs);
    } else {
      executeBatch(num_elements, inputs, outputs);
    }
  } else if (kernel->target() == KernelTarget::FPGA) {




    assert(false && "not implemented");
  }

}
// =======================================================================================================//
#ifndef SLP_DEBUG
  #define SLP_DEBUG false
#endif
// =======================================================================================================//
#if SLP_DEBUG
namespace {
  typedef std::chrono::high_resolution_clock::time_point TimePoint;
}
#endif
void Executable::executeSingle(size_t num_samples, void* inputs, void* outputs) {
  if (num_samples > 1) {
    SPDLOG_WARN("Executing a kernel optimized for single evaluation, computing only the first sample!");
  }
  assert(kernel_func);
#if SLP_DEBUG
  TimePoint start = std::chrono::high_resolution_clock::now();
#endif
  kernel_func(inputs, inputs, 0, 1, kernel->numFeatures(), 1, 1, outputs, outputs, 0, 1, 1, 1, 1);
#if SLP_DEBUG
  TimePoint end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  std::cout << "EXECUTION TIME: " << duration.count() << " ns" << std::endl;
#endif
}

void Executable::executeBatch(size_t num_samples, void* inputs, void* outputs) {
  assert(kernel_func);
  // Cast to char (i.e. byte) pointer to perform pointer arithmetic.
  char* input_ptr = reinterpret_cast<char*>(inputs);
  char* output_ptr = reinterpret_cast<char*>(outputs);
  size_t batchSize = kernel->batchSize();
#pragma omp parallel for firstprivate(input_ptr, output_ptr, batchSize, num_samples) default(none)
  for (size_t i = 0; i < num_samples; i += batchSize) {
    // Calculate the number of remaining samples, can be < batchSize for the last batch.
    size_t remainingSamples = num_samples - i;
    size_t samples = std::min(batchSize, remainingSamples);
    // Calculate pointer to first input of this batch, using information about the
    // number of features and number of bytes used to encode the feature.
    char* input_offset = &(input_ptr[i * kernel->numFeatures() * kernel->bytesPerFeature()]);
    // Calculate pointer to first output for this batch, using information about the
    // number or results and number of bytes used to encode each result.
    char* output_offset = &(output_ptr[i * kernel->numResults() * kernel->bytesPerResult()]);
    kernel_func(input_offset, input_offset, 0, samples, kernel->numFeatures(), 1, 1,
                output_offset, output_offset, 0, 1, samples, 1, 1);
  }
}

void Executable::executeGPU(size_t num_samples, void* inputs, void* outputs) {
  assert(kernel_func);
  // For GPUs, we launch all inputs at once. The host-part of the compiled kernel will split the samples
  // into multiple blocks, which in turn are processed by multiple GPU threads at once.
  // TODO: We need to specify the number of features as stride in the first dimension here,
  // as the inserted runtime calls otherwise calculate the amount of data to be tranfered wrong.
  // Investigate this further.
  kernel_func(inputs, inputs, 0, num_samples, kernel->numFeatures(), kernel->numFeatures(), 1,
              outputs, outputs, 0, 1, num_samples, 1, 1);
}

void Executable::initialize() {
  char* error = nullptr;
  bool isSharedObject = kernel->target() == KernelTarget::CPU || kernel->target() == KernelTarget::CUDA;

  if (isSharedObject) {
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
  } else {
    assert(kernel->target() == KernelTarget::FPGA);
    assert(false && "not implemented");

    kernel_func = tapasco_get_kernel_func(*kernel);

    if (!kernel_func)
      SPNC_FATAL_ERROR("Could not get kernel function in context of Tapasco");
  }
}