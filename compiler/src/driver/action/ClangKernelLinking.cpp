//
// Created by lukas on 18.01.20.
//

#include <util/Command.h>
#include "ClangKernelLinking.h"

namespace spnc {

    ClangKernelLinking::ClangKernelLinking(ActionWithOutput<ObjectFile> &_input,
                                           SharedObject outputFile, const std::string &kernelFunctionName)
                                           : ActionSingleInput<ObjectFile, SharedObject>(_input),
                                               outFile{std::move(outputFile)}, kernelName{kernelFunctionName} {}

    SharedObject& ClangKernelLinking::execute() {
        if(!cached){
            std::vector<std::string> command;
            command.emplace_back("clang");
            command.emplace_back("-shared");
            command.emplace_back("-fPIC");
            command.emplace_back("-O3");
            command.emplace_back("-o");
            command.push_back(outFile.fileName());
            command.push_back(input.execute().fileName());
            Command::executeExternalCommand(command);
            cached = true;
        }
        return outFile;
    }

}
