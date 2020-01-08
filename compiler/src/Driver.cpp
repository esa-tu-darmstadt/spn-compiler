//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <json/Parser.h>
#include <iostream>
#include <util/DotVisitor.h>
#include <transform/BinaryTreeTransform.h>
#include <codegen/llvm-ir/LLVMCodegen.h>

#define VECTORIZE true

bool spnc::parseJSON(const std::string &inputFile) {
    Parser parser;
    auto irGraph = parser.parseJSONFile(inputFile);
    //irGraph.rootNode = BinaryTreeTransform().binarizeTree(irGraph.rootNode);
    DotVisitor dot;
    dot.writeDotGraph(irGraph.rootNode, "spn.dot");
    LLVMCodegen().generateLLVMIR(irGraph, VECTORIZE);
    return true;
}
