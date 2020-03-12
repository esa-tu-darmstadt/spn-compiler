//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_IRTRANSFORMATIONPASS_H
#define SPNC_IRTRANSFORMATIONPASS_H

#include <graph-ir/GraphIRNode.h>
#include <graph-ir/GraphIRContext.h>
#include <graph-ir/IRGraph.h>
#include <driver/Actions.h>

namespace spnc {

  ///
  /// Base class for transformation passes operating on the graph-based IR.
  /// A transformation can be integrated into the toolchain as action.
  /// The transformed result is cached to avoid unnecessary recomputations.
  class IRTransformationPass : public ActionSingleInput<IRGraph, IRGraph> {

  public:

    /// Constructor.
    /// \param _input Action providing the input SPN graph.
    /// \param context GraphIRContext to manage nodes.
    explicit IRTransformationPass(ActionWithOutput<IRGraph>& _input, std::shared_ptr<GraphIRContext> context);

    IRGraph& execute() override;

    /// Execute the transformation.
    /// \param input Input SPN graph.
    virtual void transform(IRGraph& input) = 0;

  private:

    bool cached = false;

  protected:

    IRGraph transformedGraph;

  };
}




#endif //SPNC_IRTRANSFORMATIONPASS_H
