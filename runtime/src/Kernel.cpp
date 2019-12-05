//
// Created by lukas on 22.11.19.
//

#include "../include/Kernel.h"
#include <dlfcn.h>
#include <iostream>

Kernel::Kernel(const std::string &fN, const std::string &kN) : fileName{fN},
    kernelName{kN} {
  char *error;

  handle = dlopen(fileName.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Error opening kernel object file " << fileName << ": " << dlerror() << std::endl;
  }
  else{
    dlerror();    /* Clear any existing error */

    *(void **) (&kernel) = dlsym(handle, kernelName.c_str());

    if ((error = dlerror()) != NULL)  {
      std::cerr << "Could not find kernel function " << kernelName << " in " << fileName << ": " << error << std::endl;
    }
    else{
      initialized = true;
    }
  }
}


Kernel::~Kernel() {
  dlclose(handle);
}

void Kernel::execute(size_t num_elements, void *inputs, double *outputs) const {
  if(!initialized){
    std::cerr << "Cannot execute uninitialized kernel!" << std::endl;
  }
  (*kernel)(num_elements, inputs, outputs);
}
