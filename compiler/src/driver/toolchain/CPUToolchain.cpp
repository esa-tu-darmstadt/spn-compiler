//
// Created by ls on 1/15/20.
//

#include <driver/BaseActions.h>
#include <json/Parser.h>
#include <transform/BinaryTreeTransform.h>
#include <codegen/llvm-ir/CPU/LLVMCPUCodegen.h>
#include <driver/action/LLVMWriteBitcode.h>
#include "CPUToolchain.h"

namespace spnc {

    std::unique_ptr<Job<Bitcode>> CPUToolchain::constructJob(const std::string &inputFile) {
      std::unique_ptr<Job<Bitcode >> job{new Job<Bitcode>()};
      // Construct file input action.
      auto fileInput = std::make_unique<FileInputAction<FileType::SPN_JSON>>(inputFile);
      // Construct parser to parse JSON from input file.
      auto parser = std::make_unique<Parser>(*fileInput);
      // Transform all operations into binary (two inputs) operations.
      auto binaryTreeTransform = std::make_unique<BinaryTreeTransform>(*parser);
      // Invoke LLVM code-generation on transformed tree.
      auto llvmCodeGen = std::make_unique<LLVMCPUCodegen>(*binaryTreeTransform);
      // Write generated LLVM module to bitcode-file.
      auto writeBitcode = std::make_unique<LLVMWriteBitcode>(*llvmCodeGen, "test.bc");
      job->addAction(std::move(fileInput));
      job->addAction(std::move(parser));
      job->addAction(std::move(binaryTreeTransform));
      job->addAction(std::move(llvmCodeGen));
      job->setFinalAction(std::move(writeBitcode));
      return std::move(job);
    }

}