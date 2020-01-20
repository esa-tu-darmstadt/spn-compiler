//
// Created by lukas on 22.11.19.
//

#ifndef SPNC_KERNEL_H
#define SPNC_KERNEL_H

#include <optional>
#include <cstdlib>

namespace spnc {

    typedef void (*kernel_function_t)(size_t num_elements, void* inputs, double* output);

    class Kernel {

    public:

        static std::optional<Kernel> initialize(const std::string& fileName, const std::string& kernelName);

    private:

        Kernel(const std::string& fN, const std::string& kN, kernel_function_t k);

        kernel_function_t kernel;

        const std::string& fileName;

        const std::string& kernelName;

    };

}

#endif //SPNC_KERNEL_H
