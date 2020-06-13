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

    // Convert input SPN into a tree
    extern Option<bool> forceTree;
    
    enum BodyCGMethod { Scalar, ILP, Heuristic };
    // Method to use for CPU code code generation
    extern EnumOpt bodyCodeGenMethod;

    extern Option<int> simdWidth;
    /// Use AVX2 Gather instructions to load histograms
    extern Option<bool> useGather;
    /// Use select instructions instead of histograms loads for histograms with only two buckets
    extern Option<bool> selectBinary;
    // Run the ILP solver in multiple iteration of increasing node vector width
    extern Option<bool> incSolve;
    // Candidates to evaluate for whole tree
    extern Option<int> rootCand;
    // Candidates to evaluate for subtrees of vectors
    extern Option<int> depCand;
    // No. of SIMD Chains to generate
    extern Option<int> chainCandidates;
    // Factorto determine the number of SIMD Chains to generate that originate from a candidate vector
    extern Option<int> depChains;
  }
}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H
