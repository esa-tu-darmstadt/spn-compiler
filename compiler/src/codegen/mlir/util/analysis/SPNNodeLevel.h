//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNNODELEVEL_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNNODELEVEL_H

#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <frontend/json/json.hpp>
#include <mlir/IR/Module.h>
#include <util/Logging.h>
#include <set>

using json = nlohmann::json;

#include "GraphStats_enum.h"

///
/// Detail level of graph statistics.
typedef struct {
  ///
  /// Detail level.
  int level;
} GraphStatLevelInfo;

using namespace spnc;

typedef std::shared_ptr<void> arg_t;

namespace mlir {
  namespace spn {

    ///
    /// Class to walk over a (sub-)graph, logging node-levels in the process.
    /// The results may be collected via get-interfaces.
    class SPNNodeLevel {
    public:

      /// Default-Constructor.
      explicit SPNNodeLevel();

      /// Constructor.
      /// \param _root Node which will be treated as SPN-graph root.
      /// \param _rootlevel Integer which will determine the level of the provided root.
      explicit SPNNodeLevel(Operation* _root, int _rootlevel);

    private:

      /// Process provided pointer to a SPN node and update internal counts / results.
      /// \param op Pointer to the defining operation, representing a SPN node.
      /// \param arg Provided state information, e.g. (incremented) level-information from previous calls.
      void visitNode(Operation* op, const arg_t& arg);

      /// Process gathered results -- stores different depth-values which have to be calculated / determined.
      void processResults();

    public:

      /// Update (i.e. re-calculate) all node levels, starting from the root.
      void update();

      /// Return the depth of the given operation, w.r.t. the considered (sub-)graph.
      /// If the given operation could not be found, returns -1.
      /// \param op Pointer to the defining operation, representing a SPN node.
      int getDepthOperation(Operation* op) const;

      /// Return the maximum node-depth, w.r.t. the considered (sub-)graph.
      int getDepthMax() const;

      /// Return the minimum node-depth, w.r.t. the considered (sub-)graph.
      int getDepthMin() const;

      /// Return the median node-depth, w.r.t. the considered (sub-)graph.
      int getDepthMedian() const;

      /// Return the average node-depth, w.r.t. the considered (sub-)graph.
      double getDepthAvg() const;

      /// Return a copy of the full (raw) analysis result.
      std::map<Operation*, int> getResult();

      /// Return the operation (i.e. root) this analysis was constructed from.
      Operation* getRoot() const;

    private:

      Operation* root;
      int root_level = 0;

      int depth_max = 0;
      int depth_min = std::numeric_limits<int>::max();
      int depth_median = 0;

      double depth_average = 0.0;

      std::multiset<int> leaf_levels;
      std::map<Operation*, int> spn_op_levels;

    };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNNODELEVEL_H
