//
// Created by ls on 10/8/19.
//

#ifndef SPNC_PARSER_H
#define SPNC_PARSER_H

#include <unordered_map>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include "graph-ir/GraphIRNode.h"
#include "json.hpp"

using json = nlohmann::json;

namespace spnc {

class Parser : public ActionSingleInput<std::string, IRGraph> {

    public:

        explicit Parser(ActionWithOutput<std::string>& _input);

        IRGraph& execute() override ;

    private:

        IRGraph parseJSONFile(std::string& file);

        std::unordered_map<std::string, std::shared_ptr<InputVar>> inputVars;

        std::shared_ptr<GraphIRNode> parseNode(json& obj) const;

        std::shared_ptr<WeightedSum> parseSum(json& obj) const;

        std::shared_ptr<Product> parseProduct(json& obj) const;

        std::shared_ptr<Histogram> parseHistogram(json& obj) const;

        std::vector<NodeReference> parseChildren(json& obj) const;

        IRGraph graph = {nullptr, nullptr};

        bool cached = false;
    };

}




#endif //SPNC_PARSER_H
