//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNPASSES_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
    namespace low {

      std::unique_ptr<OperationPass<ModuleOp>> createLoSPNBufferizePass();

      /// Instantiate the graph stats collection pass determining SPN statistics like
      /// the number of inner and leaf nodes or min/max/average node level.
      /// \return Pass instance.
      std::unique_ptr<OperationPass<ModuleOp>> createLoSPNGraphStatsCollectionPass(const std::string& graphStatsFile);

#define GEN_PASS_REGISTRATION
#include "LoSPN/LoSPNPasses.h.inc"
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNPASSES_H
