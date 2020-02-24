//
// Created by ls on 1/14/20.
//

#ifndef SPNC_IRTRANSFORMATIONPASS_H
#define SPNC_IRTRANSFORMATIONPASS_H

#include <graph-ir/GraphIRNode.h>
#include <graph-ir/GraphIRContext.h>
#include <graph-ir/IRGraph.h>
#include <driver/Actions.h>

namespace spnc {

  class IRTransformationPass : public ActionSingleInput<IRGraph, IRGraph> {

  public:

    explicit IRTransformationPass(ActionWithOutput<IRGraph>& _input, std::shared_ptr<GraphIRContext> context);

    IRGraph& execute() override;

    virtual void transform(IRGraph& input) = 0;

  private:

    bool cached = false;

  protected:

    IRGraph transformedGraph;

  };
}




#endif //SPNC_IRTRANSFORMATIONPASS_H
