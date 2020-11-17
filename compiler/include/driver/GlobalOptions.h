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

    ///
    /// Flag to indicate whether the MLIR based toolchain should be used.
    extern Option<bool> useMLIRToolchain;

    ///
    /// Flag to indicate whether an optimal representation for SPN evaluation shall be determined.
    extern Option<bool> determineOptimalRepresentation;

    /// Available compilation targets.
    enum RepresentationOption {
      FLOATING_POINT,
      FIXED_POINT,
    };

    ///
    /// Available representation options.
    extern EnumOpt representationFormat;

    ///
    /// Flag to indicate whether the optimal representation shall be determined using the absolute or relative error.
    extern Option<bool> optimalRepresentationRelativeError;

    ///
    /// Option to set the error threshold w.r.t. the optimal representation.
    extern Option<double> optimalRepresentationErrorThreshold;

  }
}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H
