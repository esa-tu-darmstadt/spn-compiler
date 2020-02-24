//
// Created by ls on 10/8/19.
//

#include <iostream>
#include <sstream>
#include "Parser.h"

namespace spnc {

  Parser::Parser(spnc::ActionWithOutput<std::string>& _input, std::shared_ptr<GraphIRContext> context)
      : ActionSingleInput<std::string, IRGraph>{_input}, graph{context} {}

  IRGraph& Parser::execute() {
    if (!cached) {
      parseJSONFile(input.execute());
      cached = true;
    }
    return graph;
  }

  void Parser::parseJSONFile(std::string& input) {
    std::stringstream stream{input};
    json j;
    stream >> j;
    if (!j.is_object()) {
      std::cerr << "ERROR: Could not parse SPN from input string!" << std::endl;
      assert(false);
    }

    json rootNode;
    if (j.contains("Sum")) {
      rootNode = j["Sum"];
    } else if (j.contains("Product")) {
      rootNode = j["Product"];
    } else if (j.contains("Histogram")) {
      rootNode = j["Histogram"];
    } else {
      std::cerr << "Unknown root node type, could not parse top-level scope" << std::endl;
        assert(false);
      }

      if(!rootNode.contains("scope")){
        std::cerr << "No field scope in root node, could not parse top-level scope" << std::endl;
        assert(false);
      }

    json topLevelScope = rootNode["scope"];
    if (!topLevelScope.is_array()) {
      std::cerr << "Scope is not an array, could not parse top-level scope" << std::endl;
      assert(false);
    }

    size_t index = 0;
    for (json& var : topLevelScope) {
      std::string varName = var.get<std::string>();
      auto input = graph.create<InputVar>(varName, index++);
      inputVars.emplace(varName, input);
    }

    graph.setRootNode(parseNode(j));

  }

  NodeReference Parser::parseNode(json& obj) {
    if (obj.contains("Sum")) {
      return parseSum(obj);
    } else if (obj.contains("Product")) {
      return parseProduct(obj);
    } else if (obj.contains("Histogram")) {
      return parseHistogram(obj);
    } else {
      std::cerr << "Unknown node type, could not parse node" << std::endl;
      return nullptr;
    }
  }

  WeightedSum* Parser::parseSum(json& obj) {
    json body = obj["Sum"];

    if (!body.contains("id")) {
      std::cerr << "Field id missing, could not parse node" << std::endl;
      return nullptr;
    }
    auto id = body["id"].get<unsigned int>();
    if (!body.contains("weights")) {
      std::cerr << "Field weights missing, could not parse node" << std::endl;
      return nullptr;
    }
    json jsonWeights = body["weights"];

    std::vector<NodeReference> children = parseChildren(body);

    std::vector<double> weights;
    for (auto& w : jsonWeights) {
      weights.push_back(w.get<double>());
    }

    if (weights.size() != children.size()) {
      std::cerr << "Number of weights does not match number of child nodes, could not parse node" << std::endl;
      return nullptr;
    }

    std::vector<WeightedAddend> addends(0);
    auto wi = weights.begin();
    for (auto ci = children.begin(); ci != children.end(); ++ci, ++wi) {
      addends.push_back(WeightedAddend{*ci, *wi});
    }
    return graph.create<WeightedSum>(std::to_string(id), addends);
  }

  Product* Parser::parseProduct(json& obj) {
    json body = obj["Product"];

    if (!body.contains("id")) {
      std::cerr << "Field id missing, could not parse node" << std::endl;
      return nullptr;
    }
    auto id = body["id"].get<unsigned int>();

    std::vector<NodeReference> children = parseChildren(body);

    return graph.create<Product>(std::to_string(id), children);
  }

  std::vector<NodeReference> Parser::parseChildren(json& body) {
    if (!body.contains("children") || !body["children"].is_array()) {
      std::cerr << "Field children missing, could not parse node" << std::endl;
    }
    json jsonChildren = body["children"];
    std::vector<NodeReference> children(0);
    for (auto& c : jsonChildren) {
      auto childRef = parseNode(c);
      children.push_back(childRef);
    }
    return children;
  }

  Histogram* Parser::parseHistogram(json& obj) {
    json body = obj["Histogram"];

    if (!body.contains("id")) {
      std::cerr << "Field id missing, could not parse node" << std::endl;
      return nullptr;
    }
    auto id = body["id"].get<unsigned int>();

    if (!body.contains("scope") || !body["scope"].is_array() || body["scope"].size() != 1) {
      std::cerr << "Field scope not correct, could not parse node" << std::endl;
      return nullptr;
    }
    auto varName = body["scope"].at(0).get<std::string>();
    if (!inputVars.count(varName)) {
      std::cerr << "Histogram refers to unknown input variable, could not parse node" << std::endl;
      return nullptr;
    }
    auto inputVar = inputVars.at(varName);

    if (!body.contains("breaks") || !body["breaks"].is_array()) {
        std::cerr << "Field breaks missing, could not parse node" << std::endl;
        return nullptr;
      }
      json jsonBreaks = body["breaks"];

      if(!body.contains("densities") || !body["densities"].is_array()){
        std::cerr << "Field densities missing, could not parse node" << std::endl;
        return nullptr;
      }
      json jsonDensities = body["densities"];

      std::vector<int> breaks;
      for(auto& b : jsonBreaks){
        int bound = b.get<int>();
        breaks.push_back(bound);
      }

    std::vector<double> densities;
    for (auto& d : jsonDensities) {
      densities.push_back(d.get<double>());
    }
    if (breaks.size() != (densities.size() + 1)) {
      std::cerr << "Number of breaks and densities does not match, could not parse node" << std::endl;
      return nullptr;
    }

    std::vector<HistogramBucket> buckets(0);
    auto b1 = breaks.begin();
    auto b2 = breaks.begin();
    b2++;
    auto di = densities.begin();
    for (; b2 != breaks.end() && di != densities.end(); ++b1, ++b2, ++di) {
      buckets.push_back(HistogramBucket{*b1, *b2, *di});
    }
    return graph.create<Histogram>(std::to_string(id), inputVar, buckets);
  }
}

