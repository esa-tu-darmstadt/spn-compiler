//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <json/Parser.h>
#include <iostream>
#include <util/DotVisitor.h>
#include <transform/BinaryTreeTransform.h>
#include <transform/AlternatingNodesTransform.h>
#include <codegen/llvm-ir/LLVMCodegen.h>
#include "llvm/Support/CommandLine.h"

llvm::cl::OptionCategory SPNCompiler("SPN Compiler Options", "Options for controlling the spn compilation process.");

cl::opt<bool> Vectorize("vec", cl::desc("Enable vectorization"), llvm::cl::cat(SPNCompiler));


llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional, llvm::cl::Required,
					 llvm::cl::desc("<input file>"), llvm::cl::cat(SPNCompiler));

bool spnc::parseJSON(int argc, char* argv[]) {
  llvm::cl::HideUnrelatedOptions(SPNCompiler);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  Parser parser;
  auto irGraph = parser.parseJSONFile(InputFilename);
  AlternatingNodesTransform ant;
  irGraph.rootNode->accept(ant, {});
  DotVisitor dot;
  dot.writeDotGraph(irGraph.rootNode, "spn.dot");
  LLVMCodegen().generateLLVMIR(irGraph, Vectorize);
  return true;
}
