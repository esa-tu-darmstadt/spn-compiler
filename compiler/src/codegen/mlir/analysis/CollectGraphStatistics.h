//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_ANALYSIS_COLLECTGRAPHSTATISTICS_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_ANALYSIS_COLLECTGRAPHSTATISTICS_H

#include <driver/Actions.h>
#include <driver/BaseActions.h>
#include <util/json.hpp>
#include "mlir/IR/BuiltinOps.h"
#include <util/Logging.h>
#include <SPN/Analysis/SPNGraphStatistics.h>
#include <SPN/Analysis/SPNNodeLevel.h>

namespace spnc {

  ///
  /// Action to collect static graph statistics on an SPN graph in an MLIR module.
  class CollectGraphStatistics : public ActionSingleInput<mlir::ModuleOp, StatsFile> {

  public:

    explicit CollectGraphStatistics(ActionWithOutput<mlir::ModuleOp>& _input, StatsFile _statsFile);

    StatsFile& execute() override;

  private:

    void collectStatistics(mlir::ModuleOp& module);

    std::unique_ptr<mlir::spn::SPNGraphStatistics> graphStats;

    std::unique_ptr<mlir::spn::SPNNodeLevel> nodeLevel;

    StatsFile statsFile;

    bool cached = false;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_ANALYSIS_COLLECTGRAPHSTATISTICS_H
