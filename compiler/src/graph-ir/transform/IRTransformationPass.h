//
// Created by ls on 1/14/20.
//

#ifndef SPNC_IRTRANSFORMATIONPASS_H
#define SPNC_IRTRANSFORMATIONPASS_H

#include <graph-ir/GraphIRNode.h>
#include <driver/Actions.h>

namespace spnc {

    class IRTransformationPass : public ActionSingleInput<IRGraph, IRGraph> {

    public:

        explicit IRTransformationPass(ActionWithOutput<IRGraph>& _input);

        IRGraph& execute() override ;

        virtual IRGraph transform(IRGraph& input) = 0;

    private:

        bool cached = false;

        IRGraph transformedGraph;

    };
}




#endif //SPNC_IRTRANSFORMATIONPASS_H
