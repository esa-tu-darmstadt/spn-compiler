//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATSCOLLECTION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATSCOLLECTION_H

#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <codegen/mlir/util/analysis/GraphStats_enum.h>
#include <codegen/mlir/util/analysis/GraphStatsNodeCount.h>
#include <codegen/mlir/util/analysis/SPNNodeLevel.h>
#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <frontend/json/json.hpp>
#include <mlir/IR/Module.h>
#include <util/Logging.h>
#include <set>

using json = nlohmann::json;

using namespace spnc;

namespace mlir {
  namespace spn {

    ///
    /// Action to compute static graph statistics, such as node count, on the SPN graph.
    /// The result is serialized to a JSON file.
    class GraphStatsCollection : public ActionSingleInput<ModuleOp, StatsFile> {
    public:

      /// Constructor.
      /// \param _input Action providing the input SPN graph.
      /// \param outputFile File to write output to.
      explicit GraphStatsCollection(ActionWithOutput<ModuleOp>& _input, StatsFile outputFile);

      StatsFile& execute() override;

    private:

      /// Prepare SPN root and respective analyses.
      void initialize(ModuleOp&);

      /// Search module for SPNSingleQueryOp (via ReturnOp), then return the called function's (SPN) name as StringRef.
      /// \return StringRef which holds the SPN-function's name.
      StringRef getSPNFuncNameFromModule();

      /// Lookup FuncOp which returns the topmost SPN node, then return a pointer to this node.
      /// \param funcName Name of the SPN-function in the current module.
      /// \return Operation* which marks the global root of the SPN (in MLIR).
      Operation* getSPNRootByFuncName(StringRef funcName);

    public:

      /// Collect the gathered / prepared results and write to file.
      void collectGraphStats();

    private:

      int count_features = 0;
      int count_nodes_sum = 0;
      int count_nodes_product = 0;
      int count_nodes_histogram = 0;
      int count_nodes_inner = 0;
      int count_nodes_leaf = 0;
      int depth_max = 0;
      int depth_min = std::numeric_limits<int>::max();

      double depth_average = 0.0;
      double depth_median = 0.0;

      bool cached = false;

      ModuleOp* module = nullptr;

      StringRef spn_func_name = StringRef();

      Operation* spn_root_global = nullptr;

      GraphStatsNodeCount statistics_count;
      SPNNodeLevel statistics_depth;

      StatsFile outfile;
    };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_GRAPHSTATSCOLLECTION_H
