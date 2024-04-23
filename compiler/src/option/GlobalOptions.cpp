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

Option<bool> spnc::option::cpuVectorize{
    "cpu-vectorize", "Enable vectorization of generated CPU code",
    "SLP Vectorization", false};

using spnc::option::VectorLibrary;
EnumOpt spnc::option::vectorLibrary{
    "spn-vector-library",
    "Select the vector library for CPU vectorization",
    "SLP Vectorization",
    NONE,
    {EnumVal(SVML, "SVML"), EnumVal(LIBMVEC, "LIBMVEC"), EnumVal(ARM, "ARM"),
     EnumVal(NONE, "None")}};

Option<bool> spnc::option::replaceGatherWithShuffle{
    "use-shuffle", "Optimize gather loads into vector loads and shuffles",
    "Compilation", false};

Option<unsigned> spnc::option::slpMaxAttempts{
    "slp-max-attempts",
    "Set the maximum number of SLP vectorization attempts",
    "SLP Vectorization",
    1,
    {depends(spnc::option::cpuVectorize, true)}};

Option<unsigned> spnc::option::slpMaxSuccessfulIterations{
    "slp-max-successful-iterations",
    "Limit the number of successful SLP vectorization runs per function",
    "SLP Vectorization",
    1,
    {depends(spnc::option::cpuVectorize, true)}};

Option<unsigned> spnc::option::slpMaxNodeSize{
    "slp-max-node-size",
    "Define the maximum multinode size during SLP vectorization",
    "SLP Vectorization",
    10,
    {depends(spnc::option::cpuVectorize, true)}};

Option<unsigned> spnc::option::slpMaxLookAhead{
    "slp-max-look-ahead",
    "Set the maximum look-ahead depth for SLP multinode operand reordering",
    "SLP Vectorization",
    3,
    {depends(spnc::option::cpuVectorize, true)}};

Option<bool> spnc::option::slpReorderInstructionsDFS{
    "slp-reorder-dfs",
    "Choose DFS over BFS for SLP instruction reordering",
    "SLP Vectorization",
    true,
    {depends(spnc::option::cpuVectorize, true)}};

Option<bool> spnc::option::slpAllowDuplicateElements{
    "slp-allow-duplicate-elements",
    "Allow duplicate elements in vectors during SLP graph building",
    "SLP Vectorization",
    false,
    {depends(spnc::option::cpuVectorize, true)}};

Option<bool> spnc::option::slpAllowTopologicalMixing{
    "slp-allow-topological-mixing",
    "Allow elements with different topological depths in SLP vectors",
    "SLP Vectorization",
    false,
    {depends(spnc::option::cpuVectorize, true)}};

Option<bool> spnc::option::slpUseXorChains{
    "slp-use-xor-chains",
    "Use XOR chains for SLP look-ahead scores, instead of Porpodas's algorithm",
    "SLP Vectorization",
    false,
    {depends(spnc::option::cpuVectorize, true)}};

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
