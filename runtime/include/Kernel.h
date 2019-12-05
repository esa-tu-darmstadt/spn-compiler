//
// Created by lukas on 22.11.19.
//

#ifndef SPNC_KERNEL_H
#define SPNC_KERNEL_H

#include <optional>
#include <iostream>

typedef void (*kernel_function_t)(size_t num_elements, void* inputs, double* output);

class Kernel {

public:

    Kernel(std::string fN, std::string kN);

    ~Kernel();

    void execute(size_t num_elements, void* inputs, double* outputs) const;

    const std::string& fileName() const;

    const std::string& kernelName() const;

private:

    bool initialized = false;

    kernel_function_t kernel;

    void* handle;

    std::string _fileName;

    std::string _kernelName;

};

#endif //SPNC_KERNEL_H
