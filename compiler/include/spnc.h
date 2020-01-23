//
// Created by ls on 10/8/19.
//

#ifndef SPNC_SPNC_H
#define SPNC_SPNC_H

#include <string>
#include <Kernel.h>

namespace spnc {
    class spn_compiler{
    public:
        static Kernel parseJSON(const std::string& inputFile);
        Kernel parseJSONString(const std::string& jsonString);
    };
}



#endif //SPNC_SPNC_H
