//
// Created by ls on 1/20/20.
//

#ifndef SPNC_EXECUTABLE_H
#define SPNC_EXECUTABLE_H

#include <cstdlib>
#include <Kernel.h>

using namespace spnc;

namespace spnc_rt {

    typedef void (*kernel_function_t)(size_t num_elements, void* inputs, double* output);

    class Executable {

    public:
        explicit Executable(const Kernel& kernel);

        Executable(const Executable&) = delete;

        Executable& operator=(const Executable&) = delete;

        Executable(Executable&& other) noexcept ;

        Executable& operator=(Executable&& other) noexcept ;

        ~Executable();

        void execute(size_t num_elements, void* inputs, double* outputs);

    private:
        const Kernel * kernel;

        void* handle;

        kernel_function_t kernel_func;

        void initialize();

    };

}



#endif //SPNC_EXECUTABLE_H
