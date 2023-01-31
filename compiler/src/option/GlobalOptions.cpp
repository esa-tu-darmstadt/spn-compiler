//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <option/GlobalOptions.h>
#include <option/Options.h>

using namespace spnc::interface;

Option<bool> spnc::option::collectGraphStats{"collect-graph-stats", false};

Option<std::string> spnc::option::graphStatsFile{"graph-stats-file",
                                                 "/tmp/stats.json",
                                                 {depends(spnc::option::collectGraphStats, true)}};

using spnc::option::TargetMachine;
EnumOpt spnc::option::compilationTarget{"target",
                                        {EnumVal(CPU, "CPU"),
                                         EnumVal(CUDA, "CUDA"),
                                         EnumVal(FPGA, "FPGA")},
                                        {required()}};

Option<int> spnc::option::optLevel{"optLevel", 3};

Option<int> spnc::option::irOptLevel{"irOptLevel"};

Option<int> spnc::option::mcOptLevel{"mcOptLevel"};

Option<int> spnc::option::maxTaskSize{"maxTaskSize", -1};

Option<bool> spnc::option::cpuVectorize{"cpu-vectorize", false};

using spnc::option::VectorLibrary;
EnumOpt spnc::option::vectorLibrary{"vector-library", NONE,
                                    {EnumVal(SVML, "SVML"),
                                     EnumVal(LIBMVEC, "LIBMVEC"),
                                     EnumVal(ARM, "ARM"),
                                     EnumVal(NONE, "None")}};

Option<bool> spnc::option::replaceGatherWithShuffle{"use-shuffle", false};

Option<unsigned> spnc::option::slpMaxAttempts{"slp-max-attempts", 1, {depends(spnc::option::cpuVectorize, true)}};

Option<unsigned>spnc::option::slpMaxSuccessfulIterations
    {"slp-max-successful-iterations", 1, {depends(spnc::option::cpuVectorize, true)}};

Option<unsigned> spnc::option::slpMaxNodeSize{"slp-max-node-size", 10, {depends(spnc::option::cpuVectorize, true)}};

Option<unsigned> spnc::option::slpMaxLookAhead{"slp-max-look-ahead", 3, {depends(spnc::option::cpuVectorize, true)}};

Option<bool>
    spnc::option::slpReorderInstructionsDFS{"slp-reorder-dfs", true, {depends(spnc::option::cpuVectorize, true)}};

Option<bool>spnc::option::slpAllowDuplicateElements
    {"slp-allow-duplicate-elements", false, {depends(spnc::option::cpuVectorize, true)}};

Option<bool>spnc::option::slpAllowTopologicalMixing
    {"slp-allow-topological-mixing", false, {depends(spnc::option::cpuVectorize, true)}};

Option<bool>spnc::option::slpUseXorChains{"slp-use-xor-chains", false, {depends(spnc::option::cpuVectorize, true)}};

Option<bool> spnc::option::logSpace{"use-log-space", false};

Option<bool> spnc::option::gpuSharedMem{"gpu-shared-mem", true};

Option<std::string> spnc::option::searchPaths{"search-paths", ""};

Option<bool> spnc::option::deleteTemporaryFiles{"delete-temps", true};

Option<bool> spnc::option::dumpIR{"dump-ir", false};

Option<bool> spnc::option::optRepresentation{"opt-repr", false};

Option<std::string> spnc::option::fpgaFloatType{"fpga-float-type", "ufloat-31"};