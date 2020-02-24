//
// Created by lukas on 23.02.20.
//

#ifndef SPNC_COMPILER_INCLUDE_GRAPH_IR_GRAPHIRCONTEXT_H
#define SPNC_COMPILER_INCLUDE_GRAPH_IR_GRAPHIRCONTEXT_H

#include "GraphIRNode.h"
#include <type_traits>
#include <vector>
#include <memory>

namespace spnc {

  class GraphIRContext {

  public:

    template<typename N, typename ...T>
    N* create(T&& ... args) {
      static_assert(std::is_base_of<GraphIRNode, N>::value, "Must be a GraphIR node!");
      nodes.push_back(std::make_unique<N>(std::forward<T>(args)...));
      return (N*) nodes.back().get();
    }

  private:

    std::vector <std::unique_ptr<GraphIRNode>> nodes;

  };

}

#endif //SPNC_COMPILER_INCLUDE_GRAPH_IR_GRAPHIRCONTEXT_H
