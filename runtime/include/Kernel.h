//
// Created by lukas on 22.11.19.
//

#ifndef SPNC_KERNEL_H
#define SPNC_KERNEL_H

#include <optional>

typedef void (*kernel_function_t)(size_t num_elements, void* inputs, double* output);

class Kernel {

public:

    Kernel(const std::string& fN, const std::string& kN);

    ~Kernel();

    void execute(size_t num_elements, void* inputs, double* outputs) const;

private:

    bool initialized = false;

    kernel_function_t kernel;

    void* handle;

    const std::string& fileName;

    const std::string& kernelName;

};

#endif //SPNC_KERNEL_H
