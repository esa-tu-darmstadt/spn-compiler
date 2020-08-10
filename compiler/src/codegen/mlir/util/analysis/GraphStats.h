//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATS_H

#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <frontend/json/json.hpp>
#include <mlir/IR/Module.h>
#include <util/Logging.h>
#include <set>

#include "GraphStats_enum.h"

using json = nlohmann::json;

///
/// Detail level of graph statistics.
typedef struct {
  ///
  /// Detail level.
  int level;
} GraphStatLevelInfo2;

using namespace spnc;

typedef std::shared_ptr<void> arg_t;

namespace mlir {
  namespace spn {

    ///
    /// Action to compute static graph statistics, such as node count, on the SPN graph.
    /// The result is serialized to a JSON file.
    class GraphStats : public ActionSingleInput<ModuleOp, StatsFile> {
    public:

      /// Constructor.
      /// \param _input Action providing the input SPN graph.
      /// \param outputFile File to write output to.
      explicit GraphStats(ActionWithOutput<ModuleOp>& _input, StatsFile outputFile);

      StatsFile& execute() override;

    private:

      /// Search module for SPNSingleQueryOp (via ReturnOp), then return the called function's (SPN) name as StringRef.
      /// \return StringRef which holds the SPN-function's name.
      StringRef getSPNFuncNameFromModule();

      /// Lookup FuncOp which returns the topmost SPN node, then return a pointer to this node.
      /// \param funcName Name of the SPN-function in the current module.
      /// \return Operation* which marks the global root of the SPN (in MLIR).
      Operation* getSPNRootByFuncName(StringRef funcName);

      /// Process provided pointer to a SPN node and update internal counts.
      /// \param op Pointer to the defining operation, representing a SPN node.
      /// \param arg Provided state information, e.g. (incremented) level-information from previous calls.
      void visitNode(Operation* op, const arg_t& arg);

    public:

      void collectGraphStats(ModuleOp&);

    private:

      int count_features = 0;
      int count_nodes_sum = 0;
      int count_nodes_product = 0;
      int count_nodes_histogram = 0;
      int count_nodes_inner = 0;
      int count_nodes_leaf = 0;
      int depth_max = 0;
      int depth_min = std::numeric_limits<int>::max();
      int depth_median = 0;

      double depth_average = 0.0;

      bool cached = false;

      ModuleOp* module = nullptr;

      Operation* spn_root_global = nullptr;

      std::map<NODETYPE, std::multiset<int>> spn_node_stats;

      StatsFile outfile;

      StringRef spn_func_name = StringRef();
    };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATS_H
