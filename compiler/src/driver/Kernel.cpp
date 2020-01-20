//
// Created by lukas on 22.11.19.
//

#include <driver/Kernel.h>
#include <dlfcn.h>
#include <iostream>

namespace spnc {

    Kernel::Kernel(const std::string &fN, const std::string &kN, kernel_function_t k) : fileName{fN},
                                                                                        kernelName{kN}, kernel{k} {}

    std::optional<Kernel> Kernel::initialize(const std::string &fileName, const std::string &kernelName) {
        void *handle;
        kernel_function_t kernelFunc;
        char *error;

        handle = dlopen(fileName.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Error opening kernel object file " << fileName << ": " << dlerror() << std::endl;
            return {};
        }

        dlerror();    /* Clear any existing error */

        /* Writing: cosine = (double (*)(double)) dlsym(handle, "cos");
            would seem more natural, but the C99 standard leaves
            casting from "void *" to a function pointer undefined.
            The assignment used below is the POSIX.1-2003 (Technical
            Corrigendum 1) workaround; see the Rationale for the
            POSIX specification of dlsym(). */

        *(void **) (&kernelFunc) = dlsym(handle, kernelName.c_str());

        if ((error = dlerror()) != nullptr)  {
            std::cerr << "Could not find kernel function " << kernelName << " in " << fileName << ": " << error << std::endl;
            return {};
        }

        dlclose(handle);
        return Kernel(fileName, kernelName, kernelFunc);
    }

}

