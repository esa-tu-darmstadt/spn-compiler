//
// Created by lukas on 22.11.19.
//

#ifndef SPNC_KERNEL_H
#define SPNC_KERNEL_H

#include <optional>
#include <cstdlib>

namespace spnc {

    class Kernel {

    public:

        Kernel(const std::string& fN, const std::string& kN) : _fileName{fN}, _kernelName{kN} {
          _unique_id = std::hash<std::string>{}(fN + kN);
        }

        const std::string& fileName() {return _fileName;}

        const std::string& kernelName() {return _kernelName;}

        size_t unique_id(){return _unique_id;}

    private:

        const std::string _fileName;

        const std::string _kernelName;

        size_t _unique_id;

    };

}

#endif //SPNC_KERNEL_H
