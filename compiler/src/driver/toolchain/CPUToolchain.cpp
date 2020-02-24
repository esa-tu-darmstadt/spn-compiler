//
// Created by ls on 1/15/20.
//

#include <driver/BaseActions.h>
#include <json/Parser.h>
#include <transform/BinaryTreeTransform.h>
#include <codegen/llvm-ir/CPU/LLVMCPUCodegen.h>
#include <driver/action/LLVMWriteBitcode.h>
#include <driver/action/LLVMStaticCompiler.h>
#include <driver/action/ClangKernelLinking.h>
#include <util/GraphStatVisitor.h>
#include "CPUToolchain.h"

namespace spnc {

    std::unique_ptr<Job<Kernel> > CPUToolchain::constructJobFromFile(const std::string &inputFile) {
      // Construct file input action.
      auto fileInput = std::make_unique<FileInputAction<FileType::SPN_JSON>>(inputFile);
      return constructJob(std::move(fileInput));
    }

    std::unique_ptr<Job<Kernel> > CPUToolchain::constructJobFromString(const std::string &inputString) {
      // Construct string input action.
      auto stringInput = std::make_unique<StringInputAction>(inputString);
      return constructJob(std::move(stringInput));
    }

    std::unique_ptr<Job<Kernel>> CPUToolchain::constructJob(std::unique_ptr<ActionWithOutput<std::string>> input) {
      std::unique_ptr<Job<Kernel>> job{new Job<Kernel>()};
      // Construct parser to parse JSON from input.
      auto graphIRContext = std::make_shared<GraphIRContext>();
      auto& parser = job->insertAction<Parser>(*input, graphIRContext);
      // Transform all operations into binary (two inputs) operations.
      BinaryTreeTransform& binaryTreeTransform = job->insertAction<BinaryTreeTransform>(parser, graphIRContext);
      // Invoke LLVM code-generation on transformed tree.
      std::string kernelName = "spn_kernel";
      auto& llvmCodeGen = job->insertAction<LLVMCPUCodegen>(binaryTreeTransform, kernelName);
      // Collect graph statistics on transformed tree.
      // TODO: Make execution optional via configuration.
      // TODO: Determine output file-name via configuration.
      auto statsFile = FileSystem::createTempFile<FileType::STAT_JSON>(false);
      auto& graphStats = job->insertAction<GraphStatVisitor>(binaryTreeTransform, std::move(statsFile));
      // Join the two actions happening on the transformed tree (graph-stats & LLVM code-gen).
      auto& joinAction = job->insertAction<JoinAction<llvm::Module, StatsFile>>(llvmCodeGen, graphStats);
      // Write generated LLVM module to bitcode-file.
      auto bitCodeFile = FileSystem::createTempFile<FileType::LLVM_BC>();
      auto& writeBitcode = job->insertAction<LLVMWriteBitcode>(joinAction, std::move(bitCodeFile));
      // Compile generated bitcode-file to object file.
      auto objectFile = FileSystem::createTempFile<FileType::OBJECT>();
      auto& compileObject = job->insertAction<LLVMStaticCompiler>(writeBitcode, std::move(objectFile));
      // Link generated object file into shared object.
      auto sharedObject = FileSystem::createTempFile<FileType::SHARED_OBJECT>(false);
      std::cout << "Compiling to object-file " << sharedObject.fileName() << std::endl;
      auto& linkSharedObject =
          job->insertFinalAction<ClangKernelLinking>(compileObject, std::move(sharedObject), kernelName);
      job->addAction(std::move(input));
      return std::move(job);
    }

}