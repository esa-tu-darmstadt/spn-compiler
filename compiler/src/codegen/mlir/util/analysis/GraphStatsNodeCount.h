//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATSNODECOUNT_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATSNODECOUNT_H

#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <frontend/json/json.hpp>
#include <mlir/IR/Module.h>
#include <util/Logging.h>
#include <set>

using json = nlohmann::json;
using namespace spnc;

#include "GraphStats_enum.h"

namespace mlir {
  namespace spn {

    ///
    /// Class to walk over a (sub-)graph, counting nodes in the process.
    /// The results may be collected via get-interfaces.
    class GraphStatsNodeCount {
    public:

      /// Default-Constructor.
      explicit GraphStatsNodeCount();

      /// Constructor.
      /// \param _node Node which will be treated as SPN-graph root.
      explicit GraphStatsNodeCount(Operation* _node);

    private:

      /// Process provided pointer to a SPN node and update internal counts.
      /// \param op Pointer to the defining operation, representing a SPN node.
      void visitNode(Operation* op);

    public:

      /// Update (i.e. re-calculate) all node counts, starting from the root (provided at construction).
      void update();

      /// Return the count of encountered nodes with the provided type, located in the considered (sub-)graph.
      /// \param _nodetype NODETYPE enum element, representing the counted operation type.
      int getCountNodes(NODETYPE _nodetype);

      /// Return the count of inner nodes, located in the considered (sub-)graph.
      int getCountNodesInner() const;

      /// Return the count of leaf nodes, located in the considered (sub-)graph.
      int getCountNodesLeaf() const;

      /// Return a copy of the full (raw) analysis result.
      std::map<NODETYPE, int> getResult();

      /// Return the operation (i.e. root) this analysis was constructed from.
      Operation* getRoot() const;

    private:

      Operation* root;

      int count_nodes_inner = 0;
      int count_nodes_leaf = 0;

      std::map<NODETYPE, int> spn_node_counts;

    };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATSNODECOUNT_H
