//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "Kernel.h"
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
enum TargetMachine { CPU, CUDA, IPU };

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

/// Vector width
extern llvm::cl::opt<unsigned> vectorWidth;

/// Optimize gather loads into vector loads and shuffles
extern llvm::cl::opt<bool> replaceGatherWithShuffle;

/// -----------------------------------------------------------------------
/// SLP vectorization options
/// -----------------------------------------------------------------------

/// Maximum number of SLP vectorization attempts
extern llvm::cl::opt<unsigned> slpMaxAttempts;

/// Maximum number of successful SLP vectorization runs to be applied to a
/// function
extern llvm::cl::opt<unsigned> slpMaxSuccessfulIterations;

/// Maximum multinode size during SLP vectorization in terms of the number of
/// vectors they may contain
extern llvm::cl::opt<unsigned> slpMaxNodeSize;

/// Maximum look-ahead depth when reordering multinode operands during SLP
/// vectorization
extern llvm::cl::opt<unsigned> slpMaxLookAhead;

/// Flag to indicate if SLP-vectorized instructions should be arranged in DFS
/// order (true) or in BFS order (false)
extern llvm::cl::opt<bool> slpReorderInstructionsDFS;

/// Flag to indicate whether duplicate elements are allowed in vectors during
/// SLP graph building
extern llvm::cl::opt<bool> slpAllowDuplicateElements;

/// Flag to indicate if elements with different topological depths are allowed
/// in vectors during SLP graph building
extern llvm::cl::opt<bool> slpAllowTopologicalMixing;

/// Flag to indicate if XOR chains should be used to compute look-ahead scores
/// instead of Porpodas's algorithm
extern llvm::cl::opt<bool> slpUseXorChains;

/// -----------------------------------------------------------------------
/// Task partitioning options
/// -----------------------------------------------------------------------

/// Maximum number of operations per task
extern llvm::cl::opt<int> maxTaskSize;

/// -----------------------------------------------------------------------
/// IPU-specific options
/// -----------------------------------------------------------------------

/// Target IPU architecture
extern llvm::cl::opt<IPUTarget> ipuTarget;

extern llvm::cl::opt<std::string> ipuCompilerPath;

} // namespace spnc::option