//
// Created by ls on 1/16/20.
//

#include "LLVMStaticCompiler.h"
#include <util/Command.h>

namespace spnc {

    LLVMStaticCompiler::LLVMStaticCompiler(spnc::ActionWithOutput<spnc::BitcodeFile> &_input,
                                           spnc::ObjectFile outputFile)
                                           : ActionSingleInput<BitcodeFile, ObjectFile>{_input},
                                           outFile{std::move(outputFile)} {}

    ObjectFile & LLVMStaticCompiler::execute() {
      if(!cached){
        std::vector<std::string> command;
        command.emplace_back("llc");
        command.emplace_back("--relocation-model=pic");
        command.emplace_back("--filetype=obj");
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