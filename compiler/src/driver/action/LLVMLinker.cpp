//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LLVMLinker.h"
#include <util/Command.h>

namespace spnc {

  LLVMLinker::LLVMLinker(spnc::ActionWithOutput<spnc::BitcodeFile> &_input1,
                         spnc::ActionWithOutput<spnc::BitcodeFile> &_input2,
                         spnc::BitcodeFile outputFile)
                         : ActionDualInput<BitcodeFile, BitcodeFile, BitcodeFile>{_input1, _input2},
                         outFile{std::move(outputFile)} {}

    BitcodeFile & LLVMLinker::execute() {
      if(!cached){
        std::vector<std::string> command;
        command.emplace_back("llvm-link");
        command.emplace_back("-o");
        command.push_back(outFile.fileName());
        command.push_back(input1.execute().fileName());
        command.push_back(input2.execute().fileName());
        Command::executeExternalCommand(command);
        std::cout << "#out: " << outFile.fileName() << std::endl;
        std::cout << "#fn2: " << input2.execute().fileName() << std::endl;
        std::cout << "#fn1: " << input1.execute().fileName() << std::endl;
        cached = true;
      }
      return outFile;
    }

}