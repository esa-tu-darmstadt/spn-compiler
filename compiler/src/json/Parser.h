//
// Created by ls on 10/8/19.
//

#ifndef SPNC_PARSER_H
#define SPNC_PARSER_H

#include <unordered_map>
#include "../graph-ir/GraphIRNode.h"
#include "json.hpp"
#include <random>

using json = nlohmann::json;

class Parser {
public:
  Parser(): e2(rd()), dist(0.999, 1.001) {}
  IRGraph parseJSONFile(const std::string &inputFile);
private:
    std::unordered_map<std::string, std::shared_ptr<InputVar>> inputVars;

    std::shared_ptr<GraphIRNode> parseNode(json& obj);

    std::shared_ptr<WeightedSum> parseSum(json& obj);

    std::shared_ptr<Product> parseProduct(json& obj);

    std::shared_ptr<Histogram> parseHistogram(json& obj);
  
    std::shared_ptr<Gauss> parseGauss(json& obj);

    std::vector<NodeReference> parseChildren(json& obj);
    unsigned int curId = 0;
    std::random_device rd;
    std::mt19937 e2;
    std::uniform_real_distribution<> dist;
};


#endif //SPNC_PARSER_H
