//
// Created by ls on 1/15/20.
//

#include <driver/BaseActions.h>
#include <frontend/json/Parser.h>
#include <graph-ir/transform/BinaryTreeTransform.h>
#include <codegen/llvm-ir/CPU/LLVMCPUCodegen.h>
#include <driver/action/LLVMWriteBitcode.h>
#include <driver/action/LLVMStaticCompiler.h>
#include <driver/action/ClangKernelLinking.h>
#include <graph-ir/util/GraphStatVisitor.h>
#include <codegen/llvm-ir/pipeline/LLVMPipeline.h>
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
      auto parser = std::make_unique<Parser>(*input);
      // Transform all operations into binary (two inputs) operations.
      auto binaryTreeTransform = std::make_unique<BinaryTreeTransform>(*parser);
      // Invoke LLVM code-generation on transformed tree.
      std::string kernelName = "spn_kernel";
      std::shared_ptr<LLVMContext> llvmContext = std::make_shared<LLVMContext>();
      auto llvmCodeGen = std::make_unique<LLVMCPUCodegen>(*binaryTreeTransform, kernelName, llvmContext);
      // Collect graph statistics on transformed tree.
      // TODO: Make execution optional via configuration.
      // TODO: Determine output file-name via configuration.
      auto statsFile = FileSystem::createTempFile<FileType::STAT_JSON>(false);
      auto graphStats = std::make_unique<GraphStatVisitor>(*binaryTreeTransform, std::move(statsFile));
      // Join the two actions happening on the transformed tree (graph-stats & LLVM code-gen).
      auto joinAction = std::make_unique<JoinAction<llvm::Module, StatsFile>>(*llvmCodeGen, *graphStats);
      // Run LLVM IR transformation pipeline on the generated module.
      auto llvmPipeline = std::make_unique<LLVMPipeline>(*joinAction, llvmContext);
      // Write generated LLVM module to bitcode-file.
      auto bitCodeFile = FileSystem::createTempFile<FileType::LLVM_BC>();
      auto writeBitcode = std::make_unique<LLVMWriteBitcode>(*llvmPipeline, std::move(bitCodeFile));
      // Compile generated bitcode-file to object file.
      auto objectFile = FileSystem::createTempFile<FileType::OBJECT>();
      auto compileObject = std::make_unique<LLVMStaticCompiler>(*writeBitcode, std::move(objectFile));
      // Link generated object file into shared object.
      auto sharedObject = FileSystem::createTempFile<FileType::SHARED_OBJECT>(false);
      std::cout << "Compiling to object-file " << sharedObject.fileName() << std::endl;
      auto linkSharedObject = std::make_unique<ClangKernelLinking>(*compileObject, std::move(sharedObject), kernelName);
      job->addAction(std::move(input));
      job->addAction(std::move(parser));
      job->addAction(std::move(binaryTreeTransform));
      job->addAction(std::move(llvmCodeGen));
      job->addAction(std::move(graphStats));
      job->addAction(std::move(joinAction));
      job->addAction(std::move(llvmPipeline));
      job->addAction(std::move(writeBitcode));
      job->addAction(std::move(compileObject));
      job->setFinalAction(std::move(linkSharedObject));
      return std::move(job);
    }

}