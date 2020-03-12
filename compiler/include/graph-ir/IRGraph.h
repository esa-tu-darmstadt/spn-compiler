//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_INCLUDE_GRAPH_IR_IRGRAPH_H
#define SPNC_COMPILER_INCLUDE_GRAPH_IR_IRGRAPH_H

#include "GraphIRContext.h"

namespace spnc {

  ///
  /// Graph container for the graph-based IR.
  ///
  class IRGraph {

  public:

    /// Constructor.
    /// \param _context The associated GraphIRContext for managing lifetimes of the nodes.
    explicit IRGraph(std::shared_ptr<GraphIRContext> _context) : context{_context} {}

    /// Create a node in the context. This method should always be used to create GraphIRNode because it
    /// correctly manages their lifetime through the GraphIRContext.
    /// \tparam N Node type.
    /// \tparam T Types of the arguments used to construct the node.
    /// \param args Arguments used to construct the node.
    /// \return Non-owning pointer to the created node.
    template<typename N, typename ...T>
    N* create(T&& ... args) {
      auto ptr = context->create<N>(std::forward<T>(args)...);
      if (std::is_base_of<InputVar, N>::value) {
        _inputs.push_back((InputVar*) ptr);
      }
      return ptr;
    }

    /// Get the root node of this graph.
    /// \return Root node.
    NodeReference rootNode() {
      return _rootNode;
    }

    /// Set the root node of this graph.
    /// \param rootNode New root node.
    void setRootNode(NodeReference rootNode) {
      _rootNode = rootNode;
    }

    /// Get the GraphIRContext used to manage the nodes in this graph.
    /// \return Context.
    std::shared_ptr<GraphIRContext> getContext() { return context; }

    /// Get the list of features of this SPN graph.
    /// \return Input variables.
    std::vector<InputVar*>& inputs() { return _inputs; }

  private:

    std::shared_ptr<GraphIRContext> context;

    std::vector<InputVar*> _inputs;

    NodeReference _rootNode = nullptr;

  };

}

#endif //SPNC_COMPILER_INCLUDE_GRAPH_IR_IRGRAPH_H
