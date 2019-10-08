//
// Created by ls on 10/8/19.
//

#include <spnc.h>
#include <json/Parser.h>
#include <iostream>

bool spnc::parseJSON(const std::string &inputFile) {
    Parser parser;
    auto rootNode = parser.parseJSONFile(inputFile);
    std::cout << *rootNode << std::endl;
    auto root = std::static_pointer_cast<WeightedSum>(rootNode);
    for(auto& a : *root->addends()){
        std::cout << *a.addend << std::endl;
    }
    return true;
}