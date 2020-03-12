//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <util/Logging.h>
#include "Parser.h"

using namespace spnc;

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
    SPNC_FATAL_ERROR("SPN serialization does not contain a JSON object");
  }

  json rootNode;
  // Parse the root node from the top-level object.
  if (j.contains("Sum")) {
    rootNode = j["Sum"];
  } else if (j.contains("Product")) {
    rootNode = j["Product"];
  } else if (j.contains("Histogram")) {
    rootNode = j["Histogram"];
  } else {
    SPNC_FATAL_ERROR("Unknown root node type, could not parse top-level scope");
  }

  if (!rootNode.contains("scope")) {
    SPNC_FATAL_ERROR("No field 'scope' in root node, could not parse top-level scope");
  }

  json topLevelScope = rootNode["scope"];
  if (!topLevelScope.is_array()) {
    SPNC_FATAL_ERROR("Scope is not an array, could not parse top-level scope");
  }

  // The JSON serialization does not explicitly model input variables. For each entity enumerated in
  // the scope of the root-node, we create a input variable.
  size_t index = 0;
  for (json& var : topLevelScope) {
    std::string varName = var.get<std::string>();
    auto input = graph.create<InputVar>("in_" + varName, index++);
    inputVars.emplace(varName, input);
  }

  // Recurse to parse the rest of the graph and set the root node.
  graph.setRootNode(parseNode(j));

}

NodeReference Parser::parseNode(json& obj) {
  // Switch on the type of the node to parse.
  if (obj.contains("Sum")) {
    return parseSum(obj);
  } else if (obj.contains("Product")) {
    return parseProduct(obj);
  } else if (obj.contains("Histogram")) {
    return parseHistogram(obj);
  } else {
    SPNC_FATAL_ERROR("Unknown node type, could not parse node");
  }
}

WeightedSum* Parser::parseSum(json& obj) {
  json body = obj["Sum"];

  if (!body.contains("id")) {
    SPNC_FATAL_ERROR("Field id missing, could not parse node");
  }
  auto id = body["id"].get<unsigned int>();
  if (!body.contains("weights")) {
    SPNC_FATAL_ERROR("Field weights missing, could not parse node");
  }

  // Recurse to parse the child nodes.
  std::vector<NodeReference> children = parseChildren(body);

  // Try to parse the array of weights associated with the child nodes.
  json jsonWeights = body["weights"];
  std::vector<double> weights;
  for (auto& w : jsonWeights) {
    weights.push_back(w.get<double>());
  }

  if (weights.size() != children.size()) {
    SPNC_FATAL_ERROR("Number of weights does not match number of child nodes, could not parse node");
  }

  // Combine each child node and weight into WeightedAddend.
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
    SPNC_FATAL_ERROR("Field id missing, could not parse node");
  }
  auto id = body["id"].get<unsigned int>();

  // Recurse to parse child nodes.
  std::vector<NodeReference> children = parseChildren(body);

  return graph.create<Product>(std::to_string(id), children);
}

std::vector<NodeReference> Parser::parseChildren(json& body) {
  if (!body.contains("children") || !body["children"].is_array()) {
    SPNC_FATAL_ERROR("Field children missing, could not parse node");
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
    SPNC_FATAL_ERROR("Field id missing, could not parse node");
  }
  auto id = body["id"].get<unsigned int>();

  if (!body.contains("scope") || !body["scope"].is_array() || body["scope"].size() != 1) {
    SPNC_FATAL_ERROR("Field scope not correctly specified, could not parse node");
  }
  auto varName = body["scope"].at(0).get<std::string>();
  if (!inputVars.count(varName)) {
    SPNC_FATAL_ERROR("Histogram refers to unknown input variable, could not parse node");
  }
  auto inputVar = inputVars.at(varName);

  if (!body.contains("breaks") || !body["breaks"].is_array()) {
    SPNC_FATAL_ERROR("Field breaks missing, could not parse node");
  }
  json jsonBreaks = body["breaks"];

  if (!body.contains("densities") || !body["densities"].is_array()) {
    SPNC_FATAL_ERROR("Field densities missing, could not parse node");
  }
  json jsonDensities = body["densities"];

  std::vector<int> breaks;
  for (auto& b : jsonBreaks) {
    int bound = b.get<int>();
    breaks.push_back(bound);
  }

  std::vector<double> densities;
  for (auto& d : jsonDensities) {
    densities.push_back(d.get<double>());
  }
  if (breaks.size() != (densities.size() + 1)) {
    SPNC_FATAL_ERROR("Number of breaks and densities does not match, could not parse node");
  }

  // The histogram is specified by a n+1 list of integer bound and a list of n doubles.
  // The integer value at position i specifies the inclusive lower bound for the double at position i,
  // the integer value at position i+1 specifies the exclusive upper bound.
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


