//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "IRTransformationPass.h"

using namespace spnc;

IRTransformationPass::IRTransformationPass(spnc::ActionWithOutput<IRGraph>& _input,
                                           std::shared_ptr<GraphIRContext> context)
    : ActionSingleInput<IRGraph, IRGraph>(_input), transformedGraph{context} {}

IRGraph& IRTransformationPass::execute() {
  if (!cached) {
    transform(input.execute());
    cached = true;
  }
  return transformedGraph;
}
