//
// Created by ls on 10/7/19.
//

#ifndef SPNC_GRAPHIRNODE_H
#define SPNC_GRAPHIRNODE_H


#include <string>
#include <utility>
#include <vector>
#include <memory>

class GraphIRNode {

public:
    explicit GraphIRNode(std::string id);

    virtual std::string dump() const;

    friend std::ostream& operator<<(std::ostream& os, const GraphIRNode& node);

    std::string id() const;

private:
    std::string _id;

};

class InputVar : public GraphIRNode {

public:
    InputVar(std::string id, int index);

    int index() const;

    std::string dump() const override;

private:
    int _index;
};

struct HistogramBucket{int lowerBound; int upperBound; double value;};

class Histogram : public GraphIRNode {
public:
    Histogram(std::string id, InputVar* indexVar, const std::vector<HistogramBucket>& buckets);

    std::shared_ptr<InputVar> indexVar() const;

    std::shared_ptr<std::vector<HistogramBucket>> buckets() const;

private:
    std::shared_ptr<InputVar> _indexVar;

    std::shared_ptr<std::vector<HistogramBucket>> _buckets;
};

typedef std::shared_ptr<GraphIRNode> NodeReference;

struct WeightedAddend{NodeReference addend; double weight;};

class WeightedSum : public GraphIRNode {
public:
    WeightedSum(std::string id, const std::vector<WeightedAddend>& addends);

    std::shared_ptr<std::vector<WeightedAddend>> addends() const;

private:
    std::shared_ptr<std::vector<WeightedAddend>> _addends;
};

class Sum : public GraphIRNode {
public:
    Sum(std::string id, const std::vector<NodeReference>& addends);

    std::shared_ptr<std::vector<NodeReference>> addends() const;

private:
    std::shared_ptr<std::vector<NodeReference>> _addends;
};

class Product : public GraphIRNode {
public:
    Product(std::string id, const std::vector<NodeReference>& multiplicands);

    std::shared_ptr<std::vector<NodeReference>> multiplicands();

private:
    std::shared_ptr<std::vector<NodeReference>> _multiplicands;
};


#endif //SPNC_GRAPHIRNODE_H
