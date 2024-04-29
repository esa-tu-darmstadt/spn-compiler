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

Option<bool> spnc::option::collectGraphStats{
    "collect-graph-stats",
    "Enable collection of static graph statistics during compilation",
    "Compilation", false};

Option<std::string> spnc::option::graphStatsFile{
    "graph-stats-file",
    "Output file for static graph statistics",
    "Compilation",
    "/tmp/stats.json",
    {depends(spnc::option::collectGraphStats, true)}};

using spnc::option::TargetMachine;
EnumOpt spnc::option::compilationTarget{
    "target",
    "Specify the compilation target (CPU or CUDA)",
    "Compilation",
    {EnumVal(CPU, "CPU"), EnumVal(CUDA, "CUDA")},
    {required()}};

Option<int> spnc::option::optLevel{
    "opt-level", "Set the overall optimization level (0-3)", "Compilation", 3};

Option<int> spnc::option::irOptLevel{
    "irOptLevel",
    "Set the IR optimization level (0-3), overriding the general optimization "
    "level",
    "Compilation"};

Option<int> spnc::option::mcOptLevel{
    "mcOptLevel",
    "Set the machine code optimization level (0-3), overriding the general "
    "optimization level",
    "Compilation"};

Option<int> spnc::option::maxTaskSize{
    "maxTaskSize",
    "Specify the maximum size of a task for better compilation time, with "
    "potential runtime overhead",
    "Compilation", -1};

Option<bool> spnc::option::logSpace{
    "use-log-space", "Enable log-space computation for numerical stability",
    "Compilation", false};

Option<bool> spnc::option::gpuSharedMem{
    "gpu-shared-mem", "Use shared/workgroup memory for GPU computation",
    "GPU Optimization", true};

Option<std::string> spnc::option::searchPaths{
    "search-paths", "Additional search paths for libraries", "Compilation", ""};

Option<bool> spnc::option::deleteTemporaryFiles{
    "delete-temps", "Delete temporary files after compilation", "Compilation",
    true};

Option<bool> spnc::option::dumpIR{
    "dump-ir", "Dump IR after individual steps and passes", "Debug", false};

Option<bool> spnc::option::optRepresentation{
    "opt-repr",
    "Enable whether an optimial representation for SPN evaluation should be "
    "determined.",
    "Debug", false};

Option<std::string> spnc::option::debugOnly{
    "debug-only",
    "Specifies for which passes debug information should be printed", "Debug"};
