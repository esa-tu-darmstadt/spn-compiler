//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_DOTVISITOR_H
#define SPNC_DOTVISITOR_H

#include <graph-ir/transform/BaseVisitor.h>
#include <sstream>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <graph-ir/IRGraph.h>

namespace spnc {

  ///
  /// Action dumping the graph-IR into a GraphViz dot file.
  class DotVisitor : BaseVisitor, ActionSingleInput<IRGraph, File<FileType::DOT>> {

  public:

    /// Constructor.
    /// \param _input Action providing the input SPN graph.
    /// \param outputFile File to write output to.
    explicit DotVisitor(ActionWithOutput<IRGraph>& _input, const std::string& outputFile);

    File<FileType::DOT>& execute() override;

  private:

    void writeDotGraph(const NodeReference rootNode);

  public:

    void visitInputvar(InputVar& n, arg_t arg) override;

    void visitHistogram(Histogram& n, arg_t arg) override;

    void visitGauss(Gauss& n, arg_t arg) override ;

    void visitProduct(Product& n, arg_t arg) override;

    void visitSum(Sum& n, arg_t arg) override;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override;

  private:

    std::stringstream nodes{};

    std::stringstream edges{};

    bool cached = false;

    File<FileType::DOT> outfile;
  };
}

#endif //SPNC_DOTVISITOR_H
