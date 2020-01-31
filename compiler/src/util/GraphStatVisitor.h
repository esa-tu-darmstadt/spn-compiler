//
// Created by mh on 2020-JAN-27.
//

#ifndef SPNC_GRAPHSTATVISITOR_H
#define SPNC_GRAPHSTATVISITOR_H


#include "../transform/BaseVisitor.h"
#include <sstream>
#include <set>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include "../json/json.hpp"

using json = nlohmann::json;

enum class NODETYPE { SUM, PRODUCT, HISTOGRAM };

typedef struct { int level; } GraphStatLevelInfo;


namespace spnc {

    class GraphStatVisitor : BaseVisitor, ActionSingleInput<IRGraph, File<FileType::SPN_JSON>> {

    public:

        explicit GraphStatVisitor(ActionWithOutput<IRGraph>& _input, const std::string& outputFile);

        File<FileType::SPN_JSON> &execute() override;

    private:

        void collectGraphStats(const NodeReference& rootNode);

    public:

        void visitInputvar(InputVar& n, arg_t arg) override ;

        void visitHistogram(Histogram& n, arg_t arg) override ;

        void visitProduct(Product& n, arg_t arg) override ;

        void visitSum(Sum& n, arg_t arg) override ;

        void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

    private:
        int count_features = 0;
        int count_nodes_sum = 0;
        int count_nodes_product = 0;
        int count_nodes_histogram = 0;
        int count_nodes_inner = 0;
        int count_nodes_leaf = 0;
        int depth_max = 0;
        int depth_min = 0x7FFFFFFF;
        int depth_median = 0;

        double depth_average = 0.0;

        bool cached = false;

        std::map<NODETYPE, std::multimap<int, std::string>> spn_node_stats;

        File<FileType::SPN_JSON> outfile;
    };
}

#endif //SPNC_GRAPHSTATVISITOR_H
