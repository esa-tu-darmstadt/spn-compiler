#include "option/Options.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"

namespace spnc::option {

/// -----------------------------------------------------------------------
/// SPNC target options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory targetGroup{"SPNC target options"};

llvm::cl::opt<TargetMachine> compilationTarget{
    "spnc-target",
    llvm::cl::desc("Specify the compilation target (CPU or CUDA)"),
    llvm::cl::values(clEnumValN(TargetMachine::CPU, "CPU", "CPU target"),
                     clEnumValN(TargetMachine::CUDA, "CUDA", "CUDA target"),
                     clEnumValN(TargetMachine::IPU, "IPU", "IPU target")),
    llvm::cl::cat(targetGroup), llvm::cl::Required};

/// -----------------------------------------------------------------------
/// SPNC compilation options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory compilationGroup{"SPNC compilation options"};
llvm::cl::opt<std::string> searchPaths{
    "spnc-search-paths",
    llvm::cl::desc("Additional search paths for libraries"), llvm::cl::init(""),
    llvm::cl::cat(compilationGroup)};

llvm::cl::opt<bool> deleteTemporaryFiles{
    "spnc-delete-temps",
    llvm::cl::desc("Delete temporary files after compilation"),
    llvm::cl::init(true), llvm::cl::cat(compilationGroup)};

llvm::cl::opt<bool> dumpIR{
    "spnc-dump-ir", llvm::cl::desc("Dump IR after individual steps and passes"),
    llvm::cl::init(false), llvm::cl::cat(compilationGroup)};

llvm::cl::opt<bool> optRepresentation{
    "spnc-opt-repr",
    llvm::cl::desc(
        "Enable whether an optimal representation for SPN evaluation should be "
        "determined."),
    llvm::cl::init(false), llvm::cl::cat(compilationGroup)};

llvm::cl::opt<std::string> stopAfter{
    "spnc-stop-after",
    llvm::cl::desc("Stop after the specified step in the pipeline"),
    llvm::cl::init(""), llvm::cl::cat(compilationGroup)};

/// -----------------------------------------------------------------------
/// SPNC optimization options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory optimizationGroup{"SPNC optimization options"};

llvm::cl::opt<int> optLevel{
    "spnc-opt-level",
    llvm::cl::desc("Set the overall optimization level (0-3)"),
    llvm::cl::init(3), llvm::cl::cat(optimizationGroup)};

llvm::cl::opt<int> irOptLevel{
    "spnc-irOptLevel",
    llvm::cl::desc(
        "Set the IR optimization level (0-3), overriding the general "
        "optimization level"),
    llvm::cl::cat(optimizationGroup)};

llvm::cl::opt<int> mcOptLevel{
    "spnc-mcOptLevel",
    llvm::cl::desc(
        "Set the machine code optimization level (0-3), overriding the "
        "general optimization level"),
    llvm::cl::cat(optimizationGroup)};

llvm::cl::opt<bool> logSpace{
    "spnc-use-log-space",
    llvm::cl::desc("Enable log-space computation for numerical stability"),
    llvm::cl::init(false)};

llvm::cl::opt<bool> gpuSharedMem{
    "spnc-gpu-shared-mem",
    llvm::cl::desc("Use shared/workgroup memory for GPU computation"),
    llvm::cl::init(true)};

/// -----------------------------------------------------------------------
/// Statistics options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory statsCategory{"SPNC statistics options"};

llvm::cl::opt<bool> collectGraphStats{
    "spnc-collect-graph-stats",
    llvm::cl::desc("Collect static graph statistics during compilation"),
    llvm::cl::init(false), llvm::cl::cat(statsCategory)};

llvm::cl::opt<std::string> graphStatsFile{
    "spnc-graph-stats-file",
    llvm::cl::desc("Output file for static graph statistics"),
    llvm::cl::init("/tmp/stats.json"), llvm::cl::cat(statsCategory)};

/// -----------------------------------------------------------------------
/// Vectorization options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory vectorizationCategory{"SPNC vectorization options"};

using llvm::driver::VectorLibrary;
llvm::cl::opt<VectorLibrary> vectorLibrary{
    "spnc-vector-library",
    llvm::cl::desc("Specify the vector library to use for vectorization"),
    llvm::cl::values(
        clEnumValN(VectorLibrary::NoLibrary, "None", "No vector library"),
        clEnumValN(VectorLibrary::Accelerate, "Accelerate", "Accelerate"),
        clEnumValN(VectorLibrary::LIBMVEC, "LIBMVEC", "GLIBC vector math"),
        clEnumValN(VectorLibrary::MASSV, "MASSV", "IBM MASS vector library"),
        clEnumValN(VectorLibrary::SVML, "SVML", "Intel short vector math"),
        clEnumValN(VectorLibrary::SLEEF, "SLEEF",
                   "SLEEF SIMD Library for Evaluating Elementary Functions"),
        clEnumValN(VectorLibrary::Darwin_libsystem_m, "Darwin_libsystem_m",
                   "Darwin's libsystem_m vector functions"),
        clEnumValN(VectorLibrary::ArmPL, "ArmPL", "Arm Performance Libraries"),
        clEnumValN(VectorLibrary::AMDLIBM, "AMDLIBM",
                   "AMD vector math library")),
    llvm::cl::init(VectorLibrary::NoLibrary),
    llvm::cl::cat(vectorizationCategory)};

llvm::cl::opt<unsigned> vectorWidth{
    "spnc-vector-width",
    llvm::cl::desc("The vector-width to use for vectorization. Use 0 to use "
                   "the hardware vector width of the target architecture"),
    llvm::cl::init(0), llvm::cl::cat(vectorizationCategory)};

llvm::cl::opt<bool> vectorize{
    "spnc-cpu-vectorize", llvm::cl::desc("Enable vectorization for CPU target"),
    llvm::cl::init(false), llvm::cl::cat(vectorizationCategory)};

llvm::cl::opt<bool> replaceGatherWithShuffle{
    "spnc-use-shuffle",
    llvm::cl::desc("Optimize gather loads into vector loads and shuffles"),
    llvm::cl::init(false), llvm::cl::cat(vectorizationCategory)};

/// -----------------------------------------------------------------------
/// SLP vectorization options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory slpCategory{"SPNC SLP vectorization options"};

llvm::cl::opt<unsigned> slpMaxAttempts{
    "spnc-slp-max-attempts",
    llvm::cl::desc("Maximum number of SLP vectorization attempts"),
    llvm::cl::init(1), llvm::cl::cat(slpCategory)};
llvm::cl::opt<unsigned> slpMaxSuccessfulIterations{
    "spnc-slp-max-successful-iterations",
    llvm::cl::desc("Maximum number of successful SLP vectorization runs to "
                   "be applied to a function"),
    llvm::cl::init(1), llvm::cl::cat(slpCategory)};
llvm::cl::opt<unsigned> slpMaxNodeSize{
    "spnc-slp-max-node-size",
    llvm::cl::desc("Maximum multinode size during SLP vectorization in terms "
                   "of the number of vectors they may contain"),
    llvm::cl::init(10), llvm::cl::cat(slpCategory)};
llvm::cl::opt<unsigned> slpMaxLookAhead{
    "spnc-slp-max-look-ahead",
    llvm::cl::desc("Maximum look-ahead depth when reordering multinode "
                   "operands during SLP vectorization"),
    llvm::cl::init(3), llvm::cl::cat(slpCategory)};
llvm::cl::opt<bool> slpReorderInstructionsDFS{
    "spnc-slp-reorder-dfs",
    llvm::cl::desc("Flag to indicate if SLP-vectorized instructions should "
                   "be arranged in DFS order (true) or in BFS order "
                   "(false)"),
    llvm::cl::init(true), llvm::cl::cat(slpCategory)};
llvm::cl::opt<bool> slpAllowDuplicateElements{
    "spnc-slp-allow-duplicate-elements",
    llvm::cl::desc("Flag to indicate whether duplicate elements are allowed "
                   "in vectors during SLP graph building"),
    llvm::cl::init(false), llvm::cl::cat(slpCategory)};
llvm::cl::opt<bool> slpAllowTopologicalMixing{
    "spnc-slp-allow-topological-mixing",
    llvm::cl::desc("Flag to indicate if elements with different topological "
                   "depths are allowed in vectors during SLP graph "
                   "building"),
    llvm::cl::init(false), llvm::cl::cat(slpCategory)};
llvm::cl::opt<bool> slpUseXorChains{
    "spnc-slp-use-xor-chains",
    llvm::cl::desc("Flag to indicate if XOR chains should be used to "
                   "compute look-ahead scores instead of Porpodas's "
                   "algorithm"),
    llvm::cl::init(true), llvm::cl::cat(slpCategory)};

/// -----------------------------------------------------------------------
/// Task partitioning
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory partitionCategory{"SPNC task partitioning options"};

llvm::cl::opt<int> maxTaskSize{
    "spnc-max-task-size",
    llvm::cl::desc("Maximum number of operations per task"), llvm::cl::init(-1),
    llvm::cl::cat(partitionCategory)};

/// -----------------------------------------------------------------------
/// IPU options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory ipuCategory{"SPNC IPU options"};

llvm::cl::opt<IPUTarget> ipuTarget{
    "spnc-ipu-target", llvm::cl::desc("Specify the IPU target"),
    llvm::cl::values(clEnumValN(IPUTarget::IPU1, "IPU1", "IPU1 target"),
                     clEnumValN(IPUTarget::IPU2, "IPU2", "IPU2 target"),
                     clEnumValN(IPUTarget::IPU21, "IPU21", "IPU21 target"),
                     clEnumValN(IPUTarget::Model, "Model", "Model target")),
    llvm::cl::init(IPUTarget::Model), llvm::cl::cat(ipuCategory)};

llvm::cl::opt<std::string> ipuCompilerPath{
    "spnc-ipu-compiler-path",
    llvm::cl::desc("Path to the IPU compiler executable"),
    llvm::cl::init("popc"), llvm::cl::cat(ipuCategory)};
} // namespace spnc::option