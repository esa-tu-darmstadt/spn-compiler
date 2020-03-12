//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_INCLUDE_GRAPH_IR_GRAPHIRCONTEXT_H
#define SPNC_COMPILER_INCLUDE_GRAPH_IR_GRAPHIRCONTEXT_H

#include "GraphIRNode.h"
#include <type_traits>
#include <vector>
#include <memory>

namespace spnc {

  ///
  /// Context for the graph-based IR. Manages the lifetime of all graph-IR nodes.
  ///
  class GraphIRContext {

  public:

    /// Create a node in the context. This method should not be used directly, but through the interface
    /// of the IRGraph.
    /// \tparam N Node type.
    /// \tparam T Types of the arguments used to construct the node.
    /// \param args Arguments used to construct the node.
    /// \return Non-owning pointer to the created node.
    template<typename N, typename ...T>
    N* create(T&& ... args) {
      static_assert(std::is_base_of<GraphIRNode, N>::value, "Must be a GraphIR node!");
      nodes.push_back(std::make_unique<N>(std::forward<T>(args)...));
      return (N*) nodes.back().get();
    }

  private:

    std::vector<std::unique_ptr<GraphIRNode>> nodes;

  };

}

#endif //SPNC_COMPILER_INCLUDE_GRAPH_IR_GRAPHIRCONTEXT_H
