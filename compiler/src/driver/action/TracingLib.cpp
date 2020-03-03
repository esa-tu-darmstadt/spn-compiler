//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "TracingLib.h"
#include <util/Command.h>

namespace spnc {

  TracingLib::TracingLib(spnc::BitcodeFile outputFile)
                         : ActionWithOutput<BitcodeFile>(),
                         outFile{std::move(outputFile)} {}

    BitcodeFile & TracingLib::execute() {
      return outFile;
    }

}