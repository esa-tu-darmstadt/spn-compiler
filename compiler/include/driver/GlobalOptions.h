//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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

    extern Option<int> optLevel;

    extern Option<int> irOptLevel;

    extern Option<int> mcOptLevel;

    ///
    /// Option to specify the maximum size of a task. Smaller tasks typically reduce
    /// compilation time, but can introduce overhead.
    extern Option<int> maxTaskSize;

    ///
    /// Flag to indicate whether the code generated for the CPU should be vectorized.
    extern Option<bool> cpuVectorize;

    /// Available vector libraries
    enum VectorLibrary {
      ARM,
      SVML,
      LIBMVEC,
      NONE
    };

    extern EnumOpt vectorLibrary;

    extern Option<bool> replaceGatherWithShuffle;

    // SLP vectorization options.

    ///
    /// Maximum number of successful SLP vectorization iterations.
    extern Option<unsigned> slpMaxIterations;

    ///
    /// Maximum multinode size during SLP vectorization in terms of the number of vectors they may contain.
    extern Option<unsigned> slpMaxNodeSize;

    ///
    /// Maximum look-ahead depth when reordering multinode operands during SLP vectorization.
    extern Option<unsigned> slpMaxLookAhead;

    ///
    /// Flag to indicate the order in which SLP-vectorized instructions should be arranged.
    /// True to reorder them based on a depth-first search, false to reorder them based on a breadth-first search.
    extern Option<bool> slpReorderInstructionsDFS;

    ///
    /// Flag to indicate whether duplicate elements are allowed in vectors during SLP graph building.
    /// True to keep growing the graph when duplicates are encountered in a vector, false to stop growing.
    extern Option<bool> slpAllowDuplicateElements;

    ///
    /// Flag to indicate if elements with different topological depths are allowed in vectors during SLP graph building.
    /// True to allow mixing, false to stop growing the graph at that vector if mixed topological depths occur.
    extern Option<bool> slpAllowTopologicalMixing;

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

    ///
    /// Flag to indicate whether an optimal representation for SPN evaluation shall be determined.
    extern Option<bool> optRepresentation;

  }
}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H
