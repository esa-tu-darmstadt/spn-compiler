//
// Created by ls on 1/14/20.
//

#include "IRTransformationPass.h"

namespace spnc {

    IRTransformationPass::IRTransformationPass(spnc::ActionWithOutput<IRGraph>& _input)
      : ActionSingleInput<IRGraph, IRGraph>(_input){}

    IRGraph & IRTransformationPass::execute() {
      if(!cached){
        transformedGraph = transform(input.execute());
        cached = true;
      }
      return transformedGraph;
    }
}