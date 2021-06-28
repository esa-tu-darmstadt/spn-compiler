//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <driver/GlobalOptions.h>
#include <driver/Options.h>

using namespace spnc::interface;

Option<bool> spnc::option::collectGraphStats{"collect-graph-stats", false};

Option<std::string> spnc::option::graphStatsFile{"graph-stats-file",
                                                 "/tmp/stats.json",
                                                 {depends(spnc::option::collectGraphStats, true)}};

using spnc::option::TargetMachine;
EnumOpt spnc::option::compilationTarget{"target",
                                        {EnumVal(CPU, "CPU"),
                                         EnumVal(CUDA, "CUDA")},
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

Option<bool> spnc::option::logSpace{"use-log-space", false};

Option<bool> spnc::option::gpuSharedMem{"gpu-shared-mem", true};

Option<std::string> spnc::option::searchPaths{"search-paths", ""};

Option<bool> spnc::option::deleteTemporaryFiles{"delete-temps", true};

Option<bool> spnc::option::dumpIR{"dump-ir", false};

Option<bool> spnc::option::optRepresentation{"opt-repr", false};
