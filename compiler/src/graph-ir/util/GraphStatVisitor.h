//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_GRAPHSTATVISITOR_H
#define SPNC_GRAPHSTATVISITOR_H

#include "../transform/BaseVisitor.h"
#include <sstream>
#include <set>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <frontend/json/json.hpp>
#include <graph-ir/IRGraph.h>

using json = nlohmann::json;

enum class NODETYPE { SUM, PRODUCT, HISTOGRAM };

///
/// Detail level of graph statistics.
typedef struct {
  ///
  /// Detail level.
  int level;
} GraphStatLevelInfo;

namespace spnc {

  ///
  /// Action to compute static graph statistics, such as node count, on the SPN graph.
  /// The result is serialized to a JSON file.
  class GraphStatVisitor : public BaseVisitor, public ActionSingleInput<IRGraph, StatsFile> {

  public:

    /// Constructor.
    /// \param _input Action providing the input SPN graph.
    /// \param outputFile File to write output to.
    explicit GraphStatVisitor(ActionWithOutput<IRGraph>& _input, StatsFile outputFile);

    StatsFile& execute() override;

  private:

    void collectGraphStats(const NodeReference rootNode);

  public:

    void visitInputvar(InputVar& n, arg_t arg) override;

    void visitHistogram(Histogram& n, arg_t arg) override;

    void visitProduct(Product& n, arg_t arg) override;

    void visitSum(Sum& n, arg_t arg) override;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override;

  private:
    int count_features = 0;
    int count_nodes_sum = 0;
    int count_nodes_product = 0;
    int count_nodes_histogram = 0;
    int count_nodes_inner = 0;
    int count_nodes_leaf = 0;
    int depth_max = 0;
    int depth_min = std::numeric_limits<int>::max();
    int depth_median = 0;

    double depth_average = 0.0;

    bool cached = false;

    std::map<NODETYPE, std::multimap<int, std::string>> spn_node_stats;

    StatsFile outfile;
  };
}

#endif //SPNC_GRAPHSTATVISITOR_H
