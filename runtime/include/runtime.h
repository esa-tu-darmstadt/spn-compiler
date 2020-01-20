//
// Created by lukas on 22.11.19.
//

#ifndef SPNC_RUNTIME_H
#define SPNC_RUNTIME_H

#include <string>
#include <driver/Kernel.h>

namespace spnc {

    Kernel loadKernel(const std::string& fileName, const std::string& kernelName);

}

#endif //SPNC_RUNTIME_H
