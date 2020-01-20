//
// Created by ls on 10/7/19.
//

#ifndef SPNC_GRAPHIRNODE_H
#define SPNC_GRAPHIRNODE_H


#include <string>
#include <utility>
#include <vector>
#include <memory>

namespace spnc {
    /*
 * Forward declaration of types related to the visitor to break circular dependency.
 */
    class Visitor;
    typedef std::shared_ptr<void> arg_t;

    class GraphIRNode {

    public:
        explicit GraphIRNode(std::string id);

        virtual std::string dump() const;

        friend std::ostream& operator<<(std::ostream& os, const GraphIRNode& node);

        std::string id() const;

        virtual void accept(Visitor& visitor, arg_t arg) = 0;

    private:
        std::string _id;

    };

    typedef std::shared_ptr<GraphIRNode> NodeReference;

    class InputVar : public GraphIRNode {

    public:
        InputVar(std::string id, int index);

        int index() const;

        std::string dump() const override;

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
        int _index;
    };

    struct HistogramBucket{int lowerBound; int upperBound; double value;};

    class Histogram : public GraphIRNode {
    public:
        Histogram(std::string id, std::shared_ptr<InputVar> indexVar, const std::vector<HistogramBucket>& buckets);

        std::shared_ptr<InputVar> indexVar() const;

        std::shared_ptr<std::vector<HistogramBucket>> buckets() const;

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
        std::shared_ptr<InputVar> _indexVar;

        std::shared_ptr<std::vector<HistogramBucket>> _buckets;
    };

    struct WeightedAddend{NodeReference addend; double weight;};

    class WeightedSum : public GraphIRNode {
    public:
        WeightedSum(std::string id, const std::vector<WeightedAddend>& addends);

        std::shared_ptr<std::vector<WeightedAddend>> addends() const;

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
        std::shared_ptr<std::vector<WeightedAddend>> _addends;
    };

    class Sum : public GraphIRNode {
    public:
        Sum(std::string id, const std::vector<NodeReference>& addends);

        std::shared_ptr<std::vector<NodeReference>> addends() const;

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
        std::shared_ptr<std::vector<NodeReference>> _addends;
    };

    class Product : public GraphIRNode {
    public:
        Product(std::string id, const std::vector<NodeReference>& multiplicands);

        std::shared_ptr<std::vector<NodeReference>> multiplicands();

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
        std::shared_ptr<std::vector<NodeReference>> _multiplicands;
    };

    struct IRGraph{NodeReference rootNode; std::shared_ptr<std::vector<std::shared_ptr<InputVar>>> inputs;};
}



#endif //SPNC_GRAPHIRNODE_H
