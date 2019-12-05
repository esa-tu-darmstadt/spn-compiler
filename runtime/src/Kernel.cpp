//
// Created by lukas on 22.11.19.
//

#include "../include/Kernel.h"
#include <dlfcn.h>
#include <iostream>

Kernel::Kernel(std::string fN, std::string kN) : _fileName{std::move(fN)}, _kernelName{std::move(kN)} {

  char *error;

  handle = dlopen(_fileName.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Error opening kernel object file " << _fileName << ": " << dlerror() << std::endl;
  }
  else{
    dlerror();    /* Clear any existing error */

    *(void **) (&kernel) = dlsym(handle, _kernelName.c_str());

    if ((error = dlerror()) != NULL)  {
      std::cerr << "Could not find kernel function " << _kernelName << " in " << _fileName << ": " << error << std::endl;
    }
    else{
      initialized = true;
    }
  }
}

Kernel::~Kernel() {
  if(initialized){
    dlclose(handle);
  }
}

const std::string& Kernel::fileName() const{
  return _fileName;
}

const std::string& Kernel::kernelName() const{
  return _kernelName;
}

void Kernel::execute(size_t num_elements, void *inputs, double *outputs) const {
  if(!initialized){
    std::cerr << "Cannot execute uninitialized kernel!" << std::endl;
  }
  (*kernel)(num_elements, inputs, outputs);
}
