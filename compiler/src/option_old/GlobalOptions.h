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
enum TargetMachine { CPU, CUDA };

///
/// Interface option to specify the compilation target.
extern EnumOpt compilationTarget;

///
/// Flag to indicate the desired level of optimization.
/// Possible values range from 0-3
/// Is used for both, IR and machine optimization, if none of the
/// following two options is specified.
extern Option<int> optLevel;

///
/// Flag to indicate the desired level of optimization for IR optimizations.
/// Possible values range from 0-3.
/// Overrides the value specified by 'optLevel'
extern Option<int> irOptLevel;

///
/// Flag to indicate the desired level of optimization for machine
/// optimizations. Possible values range from 0-3. Overrides the value specified
/// by 'optLevel'
extern Option<int> mcOptLevel;

///
/// Option to specify the maximum size of a task. Smaller tasks typically reduce
/// compilation time, but can introduce overhead.
extern Option<int> maxTaskSize;

/// Available vector libraries
enum VectorLibrary { ARM, SVML, LIBMVEC, NONE };

///
/// Option to specify the vector library to use.
/// Possible values can be found in the enumeration above.
extern EnumOpt vectorLibrary;

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
/// Flag to indicate whether an optimal representation for SPN evaluation shall
/// be determined.
extern Option<bool> optRepresentation;

///
/// Flag to indicate for which passes debug information should be printed.
extern Option<std::string> debugOnly;

} // namespace option
} // namespace spnc

#endif // SPNC_COMPILER_INCLUDE_DRIVER_GLOBALOPTIONS_H