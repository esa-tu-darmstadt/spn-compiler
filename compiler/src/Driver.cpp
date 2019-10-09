//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <json/Parser.h>
#include <iostream>
#include <util/DotVisitor.h>

bool spnc::parseJSON(const std::string &inputFile) {
    Parser parser;
    auto rootNode = parser.parseJSONFile(inputFile);
    DotVisitor dot;
    dot.writeDotGraph(rootNode, "spn.dot");
    return true;
}