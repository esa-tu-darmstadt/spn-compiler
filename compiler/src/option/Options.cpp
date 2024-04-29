#include "option/Options.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"

namespace spnc::option {

/// -----------------------------------------------------------------------
/// SPNC target options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory targetGroup{"SPNC compilation options"};

llvm::cl::opt<TargetMachine> compilationTarget{
    "spnc-target",
    llvm::cl::desc("Specify the compilation target (CPU or CUDA)"),
    llvm::cl::values(clEnumValN(TargetMachine::CPU, "CPU", "CPU target"),
                     clEnumValN(TargetMachine::CUDA, "CUDA", "CUDA target")),
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
    llvm::cl::cat(compilationGroup)};

llvm::cl::opt<std::string> stopAfter{
    "spnc-stop-after",
    llvm::cl::desc("Stop after the specified step in the pipeline"),
    llvm::cl::init(""), llvm::cl::cat(compilationGroup)};

/// -----------------------------------------------------------------------
/// SPNC optimization options
/// -----------------------------------------------------------------------
llvm::cl::OptionCategory optimizationGroup{"SPNC compilation options"};

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
llvm::cl::OptionCategory vectorizationCategory{"SPNC Vectorization options"};

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

llvm::cl::opt<bool> vectorize{
    "spnc-cpu-vectorize", llvm::cl::desc("Enable vectorization for CPU target"),
    llvm::cl::init(false), llvm::cl::cat(vectorizationCategory)};

llvm::cl::opt<bool> replaceGatherWithShuffle{
    "spnc-use-shuffle",
    llvm::cl::desc("Optimize gather loads into vector loads and shuffles"),
    llvm::cl::init(false), llvm::cl::cat(vectorizationCategory)};

} // namespace spnc::option