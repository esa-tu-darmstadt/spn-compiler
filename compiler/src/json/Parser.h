//
// Created by ls on 10/8/19.
//

#ifndef SPNC_PARSER_H
#define SPNC_PARSER_H

#include <unordered_map>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <graph-ir/GraphIRNode.h>
#include <graph-ir/IRGraph.h>
#include <graph-ir/GraphIRContext.h>
#include "json.hpp"
#include <graph-ir/GraphIRNode.h>

using json = nlohmann::json;

namespace spnc {

  class Parser : public ActionSingleInput<std::string, IRGraph> {

  public:

    explicit Parser(ActionWithOutput<std::string>& _input, std::shared_ptr<GraphIRContext> context);

    IRGraph& execute() override;

  private:

    void parseJSONFile(std::string& file);

    std::unordered_map<std::string, InputVar*> inputVars;

    NodeReference parseNode(json& obj);

    WeightedSum* parseSum(json& obj);

    Product* parseProduct(json& obj);

    Histogram* parseHistogram(json& obj);

    std::vector<NodeReference> parseChildren(json& obj);

    IRGraph graph;

        bool cached = false;
    };

}




#endif //SPNC_PARSER_H
