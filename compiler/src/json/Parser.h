//
// Created by ls on 10/8/19.
//

#ifndef SPNC_PARSER_H
#define SPNC_PARSER_H

#include <unordered_map>
#include "../graph-ir/GraphIRNode.h"
#include "json.hpp"

using json = nlohmann::json;

class Parser {
public:
    std::shared_ptr<GraphIRNode> parseJSONFile(const std::string& inputFile);
private:
    std::unordered_map<std::string, std::shared_ptr<InputVar>> inputVars;

    std::shared_ptr<GraphIRNode> parseNode(json& obj) const;

    std::shared_ptr<WeightedSum> parseSum(json& obj) const;

    std::shared_ptr<Product> parseProduct(json& obj) const;

    std::shared_ptr<Histogram> parseHistogram(json& obj) const;

    std::vector<NodeReference> parseChildren(json& obj) const;
};


#endif //SPNC_PARSER_H
