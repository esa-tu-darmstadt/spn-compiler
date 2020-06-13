//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_PARSER_H
#define SPNC_PARSER_H

#include <unordered_map>
#include <random>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <driver/GlobalOptions.h>
#include <graph-ir/GraphIRNode.h>
#include <graph-ir/IRGraph.h>
#include <graph-ir/GraphIRContext.h>
#include "json.hpp"

using json = nlohmann::json;

namespace spnc {

  ///
  /// Parser action parsing the SPN graph from its JSON serialization.
  class Parser : public ActionSingleInput<std::string, IRGraph> {

  public:

    /// Constructor.
    /// \param _input Action providing the input JSON serialization.
    /// \param context GraphIRContext for the generated graph-IR.
    explicit Parser(ActionWithOutput<std::string> &_input,
                    std::shared_ptr<GraphIRContext> context,
                    const Configuration &_config);

    IRGraph& execute() override;

  private:

    void parseJSONFile(std::string& file);

    std::unordered_map<std::string, InputVar*> inputVars;

    NodeReference parseNode(json& obj);

    WeightedSum* parseSum(json& obj);

    Product* parseProduct(json& obj);

    Histogram* parseHistogram(json& obj);
    
    Gauss* parseGauss(json &obj);

    std::vector<NodeReference> parseChildren(json& obj);

    IRGraph graph;

    bool cached = false;

    bool forceTree;

    unsigned int curId = 0;
    
    std::random_device rd;
    std::mt19937 e2;
    std::uniform_real_distribution<> dist;
  };

}

#endif //SPNC_PARSER_H
