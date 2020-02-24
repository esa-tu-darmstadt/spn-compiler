//
// Created by lukas on 23.02.20.
//

#ifndef SPNC_COMPILER_INCLUDE_GRAPH_IR_IRGRAPH_H
#define SPNC_COMPILER_INCLUDE_GRAPH_IR_IRGRAPH_H

#include "GraphIRContext.h"

namespace spnc {

  class IRGraph {

  public:

    explicit IRGraph(std::shared_ptr<GraphIRContext> _context) : context{_context} {}

    template<typename N, typename ...T>
    N* create(T&& ... args) {
      auto ptr = context->create<N>(std::forward<T>(args)...);
      if (std::is_base_of<InputVar, N>::value) {
        _inputs.push_back((InputVar*) ptr);
      }
      return ptr;
    }

    NodeReference rootNode() {
      return _rootNode;
    }

    void setRootNode(NodeReference rootNode) {
      _rootNode = rootNode;
    }

    std::shared_ptr<GraphIRContext> getContext() { return context; }

    std::vector<InputVar*>& inputs() { return _inputs; }

  private:

    std::shared_ptr<GraphIRContext> context;

    std::vector<InputVar*> _inputs;

    NodeReference _rootNode = nullptr;

  };

}

#endif //SPNC_COMPILER_INCLUDE_GRAPH_IR_IRGRAPH_H
