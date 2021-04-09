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
      CPU,
      CUDA
    };

    ///
    /// Interface option to specify the compilation target.
    extern EnumOpt compilationTarget;

    ///
    /// Flag to indicate whether the code generated for the CPU should be vectorized.
    extern Option<bool> cpuVectorize;

    /// Available vector libraries
    enum VectorLibrary {
      SVML,
      LIBMVEC,
      NONE
    };

    extern EnumOpt vectorLibrary;

    extern Option<bool> replaceGatherWithShuffle;

    ///
    /// Flag to indicate whether log-space computation should be used.
    extern Option<bool> logSpace;

    ///
    /// Flag to indicate whether GPU computation should use shared/workgroup memory.
    extern Option<bool> gpuSharedMem;

    ///
    /// Option to pass additional search paths for libraries to the compiler.
    /// Multiple paths can be provided as colon-separated list.
    extern Option<std::string> searchPaths;

    ///
    /// Flag to indicate whether temporary files created during compilation
    /// should be deleted after the compilation completes. Defaults to true.
    extern Option<bool> deleteTemporaryFiles;

    ///
    /// Flag to enable printing of IR after the individual steps and
    /// passes in the toolchain.
    extern Option<bool> dumpIR;

  }
}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H
