//
// Created by ls on 1/15/20.
//

#include <driver/BaseActions.h>
#include <json/Parser.h>
#include <transform/BinaryTreeTransform.h>
#include <codegen/llvm-ir/CPU/LLVMCPUCodegen.h>
#include <driver/action/LLVMWriteBitcode.h>
#include <driver/action/LLVMStaticCompiler.h>
#include "CPUToolchain.h"

namespace spnc {

    std::unique_ptr<Job<ObjectFile>> CPUToolchain::constructJob(const std::string &inputFile) {
      std::unique_ptr<Job<ObjectFile>> job{new Job<ObjectFile >()};
      // Construct file input action.
      auto fileInput = std::make_unique<FileInputAction<FileType::SPN_JSON>>(inputFile);
      // Construct parser to parse JSON from input file.
      auto parser = std::make_unique<Parser>(*fileInput);
      // Transform all operations into binary (two inputs) operations.
      auto binaryTreeTransform = std::make_unique<BinaryTreeTransform>(*parser);
      // Invoke LLVM code-generation on transformed tree.
      std::string kernelName = "spn_kernel";
      auto llvmCodeGen = std::make_unique<LLVMCPUCodegen>(*binaryTreeTransform, kernelName);
      // Write generated LLVM module to bitcode-file.
      auto bitCodeFile = FileSystem::createTempFile<FileType::LLVM_BC>();
      auto writeBitcode = std::make_unique<LLVMWriteBitcode>(*llvmCodeGen, std::move(bitCodeFile));
      // Compile generated bitcode-file to object file.
      auto objectFile = FileSystem::createTempFile<FileType::OBJECT>(false);
      std::cout << "Compiling to object-file " << objectFile.fileName() << std::endl;
      auto compileObject = std::make_unique<LLVMStaticCompiler>(*writeBitcode, std::move(objectFile));
      job->addAction(std::move(fileInput));
      job->addAction(std::move(parser));
      job->addAction(std::move(binaryTreeTransform));
      job->addAction(std::move(llvmCodeGen));
      job->addAction(std::move(writeBitcode));
      job->setFinalAction(std::move(compileObject));
      return std::move(job);
    }

}