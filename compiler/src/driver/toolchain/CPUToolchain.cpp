//
// Created by ls on 1/15/20.
//

#include <driver/BaseActions.h>
#include <frontend/json/Parser.h>
#include <graph-ir/transform/BinaryTreeTransform.h>
#include <codegen/llvm-ir/CPU/LLVMCPUCodegen.h>
#include <driver/action/LLVMWriteBitcode.h>
#include <driver/action/LLVMStaticCompiler.h>
#include <driver/action/LLVMLinker.h>
#include <driver/action/DetectTracingLib.h>
#include <driver/action/ClangKernelLinking.h>
#include <graph-ir/util/GraphStatVisitor.h>
#include <codegen/llvm-ir/pipeline/LLVMPipeline.h>
#include <driver/GlobalOptions.h>
#include "CPUToolchain.h"

namespace spnc {

  std::unique_ptr<Job<Kernel> > CPUToolchain::constructJobFromFile(const std::string& inputFile,
                                                                   const Configuration& config) {
    // Construct file input action.
    auto fileInput = std::make_unique<FileInputAction>(inputFile);
    return constructJob(std::move(fileInput), config);
  }

  std::unique_ptr<Job<Kernel> > CPUToolchain::constructJobFromString(const std::string& inputString,
                                                                     const Configuration& config) {
    // Construct string input action.
    auto stringInput = std::make_unique<StringInputAction>(inputString);
    return constructJob(std::move(stringInput), config);
  }

  std::unique_ptr<Job<Kernel>> CPUToolchain::constructJob(std::unique_ptr<ActionWithOutput<std::string>> input,
                                                          const Configuration& config) {
    auto deleteTmps = spnc::option::deleteTemporaryFiles.get(config);
    std::unique_ptr<Job<Kernel>> job{new Job<Kernel>()};
    // Construct parser to parse JSON from input.
    auto graphIRContext = std::make_shared<GraphIRContext>();
    auto& parser = job->insertAction<Parser>(*input, graphIRContext);
    // Transform all operations into binary (two inputs) operations.
    auto& binaryTreeTransform = job->insertAction<BinaryTreeTransform>(parser, graphIRContext);
    // Invoke LLVM code-generation on transformed tree.
    std::string kernelName = "spn_kernel";
    std::shared_ptr<LLVMContext> llvmContext = std::make_shared<LLVMContext>();
    auto& llvmCodeGen = job->insertAction<LLVMCPUCodegen>(binaryTreeTransform, kernelName, llvmContext);
    ActionWithOutput<llvm::Module>* codegenResult = &llvmCodeGen;
    // If requested via the configuration, collect graph statistics.
    if (spnc::option::collectGraphStats.get(config)) {
      // Collect graph statistics on transformed tree.
      auto statsFile = StatsFile(spnc::option::graphStatsFile.get(config), deleteTmps);
      auto& graphStats = job->insertAction<GraphStatVisitor>(binaryTreeTransform, std::move(statsFile));
      // Join the two actions happening on the transformed tree (graph-stats & LLVM code-gen).
      auto& joinAction = job->insertAction<JoinAction<llvm::Module, StatsFile>>(llvmCodeGen, graphStats);
      codegenResult = &joinAction;
    }
    // Run LLVM IR transformation pipeline on the generated module.
    auto& llvmPipeline = job->insertAction<LLVMPipeline>(*codegenResult, llvmContext, config);
    // Write generated LLVM module to bitcode-file.
    auto bitCodeFile = FileSystem::createTempFile<FileType::LLVM_BC>(deleteTmps);
    auto& writeBitcode = job->insertAction<LLVMWriteBitcode>(llvmPipeline, std::move(bitCodeFile));
    ActionWithOutput<BitcodeFile>* bitcode = &writeBitcode;
    if (spnc::option::numericalTracing.get(config)) {
      // Link tracing library (bitcode) into prepared bitcode-file (which yields another bitcode-file)
      auto bitCodeFileLinked = FileSystem::createTempFile<FileType::LLVM_BC>(deleteTmps);
      auto& bitCodeTraceLib = job->insertAction<DetectTracingLib>();
      auto& linkBitcode = job->insertAction<LLVMLinker>(writeBitcode, bitCodeTraceLib, std::move(bitCodeFileLinked));
      bitcode = &linkBitcode;
    }
    // Compile generated bitcode-file to object file.
    auto objectFile = FileSystem::createTempFile<FileType::OBJECT>(deleteTmps);
    auto& compileObject = job->insertAction<LLVMStaticCompiler>(*bitcode, std::move(objectFile));
    // Link generated object file into shared object.
    auto sharedObject = FileSystem::createTempFile<FileType::SHARED_OBJECT>(false);
    std::cout << "Compiling to object-file " << sharedObject.fileName() << std::endl;
    auto& linkSharedObject =
        job->insertFinalAction<ClangKernelLinking>(compileObject, std::move(sharedObject), kernelName);
    job->addAction(std::move(input));
    return std::move(job);
  }

}