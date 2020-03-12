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
  other.kernel_func = nullptr;
  other.handle = nullptr;
  other.kernel = nullptr;
}

Executable& Executable::operator=(Executable&& other) noexcept {
  kernel_func = other.kernel_func;
  other.kernel_func = nullptr;
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
  kernel_func(num_elements, inputs, outputs);
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
    SPNC_FATAL_ERROR("Could not located Kernel function {} in {}: {}", kernel->kernelName(), kernel->fileName(), error);
  }
}