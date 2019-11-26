//
// Created by lukas on 22.11.19.
//

#ifndef SPNC_RUNTIME_H
#define SPNC_RUNTIME_H

#include <string>
#include "Kernel.h"

Kernel loadKernel(const std::string& fileName, const std::string& kernelName);

#endif //SPNC_RUNTIME_H
