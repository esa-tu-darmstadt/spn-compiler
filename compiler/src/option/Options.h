//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/CommandLine.h"

#pragma once

///
/// Namespace for all configuration interface options
///
namespace spnc::option {

/// -----------------------------------------------------------------------
/// SPNC target options
/// -----------------------------------------------------------------------

/// Available compilation targets.
enum TargetMachine { CPU, CUDA };

/// Specifies the compilation target (CPU or CUDA)
extern llvm::cl::opt<TargetMachine> compilationTarget;

/// -----------------------------------------------------------------------
/// SPNC compilation options
/// -----------------------------------------------------------------------

/// Additional search paths for libraries
extern llvm::cl::opt<std::string> searchPaths;

/// Flag to indicate whether temporary files created during compilation
/// should be deleted after the compilation completes.
extern llvm::cl::opt<bool> deleteTemporaryFiles;

/// Flag to enable printing of IR after the individual steps and
/// passes in the toolchain.
extern llvm::cl::opt<bool> dumpIR;

/// Flag to indicate whether an optimal representation for SPN evaluation shall
/// be determined.
extern llvm::cl::opt<bool> optRepresentation;

/// Option to specify a step after which the
/// compilation pipeline should be terminated prematurely.
extern llvm::cl::opt<std::string> stopAfter;

/// -----------------------------------------------------------------------
/// SPNC optimization options
/// -----------------------------------------------------------------------

/// Set the overall optimization level (0-3)
extern llvm::cl::opt<int> optLevel;

/// Set the optimization level for IR optimizations (0-3)
extern llvm::cl::opt<int> irOptLevel;

/// Set the optimization level for machine optimizations (0-3)
extern llvm::cl::opt<int> mcOptLevel;

/// Enable log-space computation for numerical stability
extern llvm::cl::opt<bool> logSpace;

/// Use shared/workgroup memory for GPU computation
extern llvm::cl::opt<bool> gpuSharedMem;

/// -----------------------------------------------------------------------
/// Statistics options
/// -----------------------------------------------------------------------

/// Flag to enable the collection of graph statistics
extern llvm::cl::opt<bool> collectGraphStats;

/// File to write the graph statistics to
extern llvm::cl::opt<std::string> graphStatsFile;

/// -----------------------------------------------------------------------
/// Vectorization options
/// -----------------------------------------------------------------------

/// Vector library to use for vectorization
extern llvm::cl::opt<llvm::driver::VectorLibrary> vectorLibrary;

/// Enable vectorization
extern llvm::cl::opt<bool> vectorize;

/// Optimize gather loads into vector loads and shuffles
extern llvm::cl::opt<bool> replaceGatherWithShuffle;

} // namespace spnc::option