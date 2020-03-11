//
// Created by ls on 10/8/19.
//

#ifndef SPNC_SPNC_H
#define SPNC_SPNC_H

#include <string>
#include <Kernel.h>
#include <map>

namespace spnc {

  using options_t = std::map<std::string, std::string>;

  class spn_compiler {
  public:
    static Kernel parseJSON(const std::string& inputFile, const options_t& options);
    Kernel parseJSONString(const std::string& jsonString, const options_t& options);

  private:

    static void initLogger();

    static bool initOnce;

  };
}



#endif //SPNC_SPNC_H
