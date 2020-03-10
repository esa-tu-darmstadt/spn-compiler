//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H
#define SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H

#include "Options.h"

using namespace spnc::interface;

namespace spnc {
  namespace option {

    extern Option<bool> collectGraphStats;
    extern Option<std::string> graphStatsFile;

    enum TargetMachine {
      CPU
    };

    extern EnumOpt compilationTarget;

    extern Option<bool> deleteTemporaryFiles;

  }
}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H
