//
// Created by ls on 10/8/19.
//

#include <fstream>
#include <iostream>
#include "Parser.h"

#include "llvm/Support/CommandLine.h"

extern llvm::cl::OptionCategory SPNCompiler;

llvm::cl::opt<bool> forceTree(
    "forceTree",
    llvm::cl::desc("Even if the SPN is not a tree, duplicate nodes such that it is."),
    llvm::cl::cat(SPNCompiler));

IRGraph Parser::parseJSONFile(const std::string& inputFile) {
    std::cout << "Input file: " << inputFile << std::endl;
    std::ifstream i(inputFile);
    // TODO Check that opening the file worked out;
    json j;
    i >> j;
    if(!j.is_object()){
        std::cerr << "ERROR: Could not parse SPN from " << inputFile << std::endl;
        assert(false);
    }

    json rootNode;
    if(j.contains("Sum")){
        rootNode = j["Sum"];
    }
    else if(j.contains("Product")){
        rootNode = j["Product"];
    }
    else if(j.contains("Histogram")){
        rootNode = j["Histogram"];
    }
    else if(j.contains("Gaussian")){
        rootNode = j["Gaussian"];
    }
    else{
        std::cerr << "Unknown root node type, could not parse top-level scope" << std::endl;
        assert(false);
    }

    if(!rootNode.contains("scope")){
        std::cerr << "No field scope in root node, could not parse top-level scope" << std::endl;
        assert(false);
    }

    json topLevelScope = rootNode["scope"];
    if(!topLevelScope.is_array()){
        std::cerr << "Scope is not an array, could not parse top-level scope" << std::endl;
        assert(false);
    }

    auto inputs = std::make_shared<std::vector<std::shared_ptr<InputVar>>>(0);
    size_t index = 0;
    for(json& var : topLevelScope){
        std::string varName = var.get<std::string>();
        auto input = std::make_shared<InputVar>(varName, index++);
        inputVars.emplace(varName, input);
        inputs->push_back(input);
    }

    return IRGraph{parseNode(j), inputs};
}

std::shared_ptr<GraphIRNode> Parser::parseNode(json &obj)  {
    if(obj.contains("Sum")){
        return parseSum(obj);
    }
    else if(obj.contains("Product")){
        return parseProduct(obj);
    }
    else if(obj.contains("Histogram")){
        return parseHistogram(obj);
    }
    else if(obj.contains("Gaussian")){
        return parseGauss(obj);
    }
    else{
        std::cerr << "Unknown node type, could not parse node" << std::endl;
        return nullptr;
    }
}

std::shared_ptr<WeightedSum> Parser::parseSum(json &obj) {
    json body = obj["Sum"];
    unsigned int id;
    if (forceTree) {
      id = curId;
      curId++;
    } else {
      if (!body.contains("id")) {
        std::cerr << "Field id missing, could not parse node" << std::endl;
        return nullptr;
      }
      id = body["id"].get<unsigned int>();
    }
    if(!body.contains("weights")){
        std::cerr << "Field weights missing, could not parse node" << std::endl;
        return nullptr;
    }
    json jsonWeights = body["weights"];

    std::vector<NodeReference> children = parseChildren(body);

    std::vector<double> weights;
    for(auto& w : jsonWeights){
        weights.push_back(w.get<double>());
    }

    if(weights.size() != children.size()){
        std::cerr << "Number of weights does not match number of child nodes, could not parse node" << std::endl;
        return nullptr;
    }

    std::vector<WeightedAddend> addends;
    auto wi = weights.begin();
    for(auto ci = children.begin(); ci != children.end(); ++ci, ++wi){
        addends.push_back(WeightedAddend{*ci, *wi});
    }
    return std::make_shared<WeightedSum>(std::to_string(id), addends);
}

std::shared_ptr<Product> Parser::parseProduct(json &obj)  {
    json body = obj["Product"];

    unsigned int id;
    if (forceTree) {
      id = curId;
      curId++;
    } else {
      if (!body.contains("id")) {
        std::cerr << "Field id missing, could not parse node" << std::endl;
        return nullptr;
      }
      id = body["id"].get<unsigned int>();
    }

    std::vector<NodeReference> children = parseChildren(body);

    return std::make_shared<Product>(std::to_string(id), children);
}

std::vector<NodeReference> Parser::parseChildren(json &body)  {
    if(!body.contains("children") || !body["children"].is_array()){
        std::cerr << "Field children missing, could not parse node" << std::endl;
    }
    json jsonChildren = body["children"];
    std::vector<NodeReference> children;
    for(auto& c : jsonChildren){
        auto childRef = parseNode(c);
        children.push_back(childRef);
    }
    return children;
}


std::shared_ptr<Gauss> Parser::parseGauss(json &obj)  {
    json body = obj["Gaussian"];

    unsigned int id;
    if (forceTree) {
      id = curId;
      curId++;
    } else {
      if (!body.contains("id")) {
        std::cerr << "Field id missing, could not parse node" << std::endl;
        return nullptr;
      }
      id = body["id"].get<unsigned int>();
    }

    if(!body.contains("scope") || !body["scope"].is_array() || body["scope"].size()!=1){
        std::cerr << "Field scope not correct, could not parse node" << std::endl;
        return nullptr;
    }
    auto varName = body["scope"].at(0).get<std::string>();
    if(!inputVars.count(varName)){
        std::cerr << "Gaussian refers to unknown input variable, could not parse node" << std::endl;
        return nullptr;
    }
    auto inputVar = inputVars.at(varName);

    if(!body.contains("mean")){
        std::cerr << "Field mean missing, could not parse node" << std::endl;
        return nullptr;
    }
    auto mean = body["mean"].get<double>();

    if (forceTree) {
      double randomNum = dist(e2);
      mean = mean*randomNum;
    }

    
    if(!body.contains("stdev")){
        std::cerr << "Field stddev missing, could not parse node" << std::endl;
        return nullptr;
    }
    auto stddev = body["stdev"].get<double>();
    return std::make_shared<Gauss>(std::to_string(id), inputVar, mean, stddev);
}

std::shared_ptr<Histogram> Parser::parseHistogram(json &obj)  {
    json body = obj["Histogram"];

    unsigned int id;
    if (forceTree) {
      id = curId;
      curId++;
    } else {
      if (!body.contains("id")) {
        std::cerr << "Field id missing, could not parse node" << std::endl;
        return nullptr;
      }
      id = body["id"].get<unsigned int>();
    }

    if(!body.contains("scope") || !body["scope"].is_array() || body["scope"].size()!=1){
        std::cerr << "Field scope not correct, could not parse node" << std::endl;
        return nullptr;
    }
    auto varName = body["scope"].at(0).get<std::string>();
    if(!inputVars.count(varName)){
        std::cerr << "Histogram refers to unknown input variable, could not parse node" << std::endl;
        return nullptr;
    }
    auto inputVar = inputVars.at(varName);

    if(!body.contains("breaks") || !body["breaks"].is_array()){
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
    for(auto& d : jsonDensities){
        densities.push_back(d.get<double>());
    }
    if(breaks.size()!=(densities.size()+1)){
        std::cerr << "Number of breaks and densities does not match, could not parse node" << std::endl;
        return nullptr;
    }

    std::vector<HistogramBucket> buckets;
    auto b1 = breaks.begin();
    auto b2 = breaks.begin();
    b2++;
    auto di = densities.begin();
    for(; b2!=breaks.end() && di != densities.end(); ++b1, ++b2, ++di){
        buckets.push_back(HistogramBucket{*b1, *b2, *di});
    }
    return std::make_shared<Histogram>(std::to_string(id), inputVar, buckets);
}
