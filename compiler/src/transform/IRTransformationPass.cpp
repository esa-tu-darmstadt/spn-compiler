//
// Created by ls on 1/14/20.
//

#include "IRTransformationPass.h"

namespace spnc {

  IRTransformationPass::IRTransformationPass(spnc::ActionWithOutput<IRGraph>& _input,
                                             std::shared_ptr <GraphIRContext> context)
      : ActionSingleInput<IRGraph, IRGraph>(_input), transformedGraph{context} {}

  IRGraph& IRTransformationPass::execute() {
    if (!cached) {
      transform(input.execute());
      cached = true;
    }
    return transformedGraph;
  }
}