//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "CodeGenVecBody.h"
#include "vec/IREmitter.h"
#include <unordered_set>

using namespace spnc;


Value* CodeGenVecBody::emitBody(IRGraph& graph, Value* indVar, InputVarValueMap inputs, OutputAddressMap output, const Configuration& config) {
    std::unordered_map<std::string, size_t> partOf;
    std::unordered_map<size_t, std::unordered_set<size_t>> directVecInputs;
    std::vector<std::vector<NodeReference>> vectors;
    size_t usedSIMDWidth = spnc::option::simdWidth.get(config);
    IREmitter codeEmitter(partOf, directVecInputs, vectors, inputs, &function,
                          function.getContext(), builder, &module, usedSIMDWidth, indVar, config);

    for (auto inputVar : graph.inputs()) {
      codeEmitter.input2value[inputVar->id()] = inputs(inputVar->index(), indVar);
    }
    graph.rootNode()->accept(codeEmitter, {});
    auto storeAddress = output(indVar);
    builder.CreateStore(codeEmitter.getNodeMap()[graph.rootNode()->id()].val, storeAddress);
    std::cout << "ran iremitter " << std::endl;
    // Increment the loop induction variable.
    auto const1 = ConstantInt::get(indVar->getType(), 1);
    return builder.CreateAdd(indVar, const1, "indvar.incr");
}
