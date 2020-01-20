//
// Created by ls on 1/20/20.
//

#include <dlfcn.h>
#include <iostream>
#include "Executable.h"

namespace spnc_rt {

    Executable::Executable(const Kernel& _kernel) : kernel{&_kernel}, handle{nullptr}, kernel_func{nullptr} {}

    Executable::Executable(spnc_rt::Executable &&other) noexcept : kernel{other.kernel}, handle{other.handle},
      kernel_func{other.kernel_func} {
      other.kernel_func = nullptr;
      other.handle = nullptr;
      other.kernel = nullptr;
    }

    Executable & Executable::operator=(Executable &&other) noexcept {
      kernel_func = other.kernel_func;
      other.kernel_func = nullptr;
      handle = other.handle;
      other.handle = nullptr;
      kernel = other.kernel;
      other.kernel = nullptr;
      return *this;
    }

    Executable::~Executable(){
      if(handle){
        dlclose(handle);
      }
    }

    void Executable::execute(size_t num_elements, void *inputs, double *outputs) {
      if(!handle){
        initialize();
      }
      kernel_func(num_elements, inputs, outputs);
    }

    void Executable::initialize() {
      char* error;
      handle = dlopen(kernel->fileName().c_str(), RTLD_LAZY);
      if(!handle) {
        std::cerr << "Error opening kernel object file " << kernel->fileName() << ": " << dlerror() << std::endl;
        throw std::system_error{};
      }

      // Clear existing errors.
      dlerror();

      *(void**) (&kernel_func) = dlsym(handle, kernel->kernelName().c_str());

      if((error = dlerror()) != nullptr){
        std::cerr << "Could not find kernel function " << kernel->kernelName();
        std::cerr << " in " << kernel->fileName() << ": " << error << std::endl;
        throw std::system_error{};
      }
    }
}