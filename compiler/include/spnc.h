//
// Created by ls on 10/8/19.
//

#ifndef SPNC_SPNC_H
#define SPNC_SPNC_H

#include <string>

namespace spnc {
    class spn_compiler{
    public:
        static bool parseJSON(const std::string& inputFile);
        bool parseJSONString(const std::string& jsonString);
    };
}



#endif //SPNC_SPNC_H
