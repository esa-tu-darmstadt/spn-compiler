//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H
#define SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H

#include "Options.h"

using namespace spnc::interface;

namespace spnc {

  ///
  /// Namespace for all configuration interface options.
  ///
  namespace option {

    ///
    /// Flag to indicate whether static graph statistics (e.g. number of nodes)
    /// should be collected during compilation.
    extern Option<bool> collectGraphStats;
    ///
    /// Output-file for the static graph-statistics. Defaults to a temporary file.
    extern Option<std::string> graphStatsFile;

    /// Available compilation targets.
    enum TargetMachine {
      CPU
    };

    ///
    /// Interface option to specify the compilation target.
    extern EnumOpt compilationTarget;

    ///
    /// Flag to indicate whether temporary files created during compilation
    /// should be deleted after the compilation completes. Defaults to true.
    extern Option<bool> deleteTemporaryFiles;

  }
}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H
