//
// Created by ls on 1/17/20.
//

#ifndef SPNC_COMMAND_H
#define SPNC_COMMAND_H

#include <vector>
#include <sstream>

namespace spnc {

    class Command {

    public:

        static int executeExternalCommand(const std::string& command){
          if(!system(nullptr)){
            std::cerr << "ERROR: No command processor available!" << std::endl;
            throw std::system_error{};
          }
          return system(command.c_str());
        }

        static int executeExternalCommand(const std::vector<std::string>& command){
          std::ostringstream oss;
          for(auto& c : command){
            oss << c << " ";
          }
          int ret =  executeExternalCommand(oss.str());
          if(ret){
            std::cerr << "Error executing external command: " << ret << std::endl;
            throw std::system_error{};
          }
          return ret;
        }

    private:
        explicit Command() = default;

    };

}

#endif //SPNC_COMMAND_H
