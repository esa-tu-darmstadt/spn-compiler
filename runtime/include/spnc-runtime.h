//
// Created by lukas on 22.11.19.
//

#ifndef SPNC_RUNTIME_H
#define SPNC_RUNTIME_H

// Convenience header to include spnc runtime functionality with a single include with a meaningful name.

#include <string>
#include <Kernel.h>

using namespace spnc;

namespace spnc_rt {

    class spn_runtime {

    public:

        static spn_runtime& instance();

        void execute(const Kernel& kernel, size_t num_elements, void* inputs, double* outputs);

        spn_runtime(const spn_runtime&) = delete;

        spn_runtime(spn_runtime&&) = delete;

        spn_runtime& operator=(const spn_runtime&) = delete;

        spn_runtime& operator=(spn_runtime&&) = delete;

    private:

        explicit spn_runtime() = default;

        static spn_runtime* _instance;

    };



}

#endif //SPNC_RUNTIME_H
