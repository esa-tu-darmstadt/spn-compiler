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

      virtual ~GraphIRNode() {}

    private:
      std::string _id;

    };

  using NodeReference = GraphIRNode*;

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
      Histogram(std::string id, InputVar* indexVar, const std::vector <HistogramBucket>& buckets);

      InputVar& indexVar() const;

      const std::vector <HistogramBucket>& buckets() const;

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
      InputVar* _indexVar;

      std::vector <HistogramBucket> _buckets;
    };

    struct WeightedAddend{NodeReference addend; double weight;};

    class WeightedSum : public GraphIRNode {
    public:
        WeightedSum(std::string id, const std::vector<WeightedAddend>& addends);

      const std::vector <WeightedAddend>& addends() const;

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
      std::vector <WeightedAddend> _addends;
    };

    class Sum : public GraphIRNode {
    public:
        Sum(std::string id, const std::vector<NodeReference>& addends);

      const std::vector <NodeReference>& addends() const;

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
      std::vector <NodeReference> _addends;
    };

    class Product : public GraphIRNode {
    public:
        Product(std::string id, const std::vector<NodeReference>& multiplicands);

      const std::vector <NodeReference>& multiplicands() const;

        void accept(Visitor& visitor, arg_t arg) override ;

    private:
      std::vector <NodeReference> _multiplicands;
    };

}



#endif //SPNC_GRAPHIRNODE_H
