//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNPASSES_H
#define SPNC_MLIR_INCLUDE_DIALECT_LOSPN_LOSPNPASSES_H

#include "mlir/Pass/Pass.h"
#include "LoSPNOps.h"

namespace mlir {
  namespace spn {
    namespace low {

      std::unique_ptr<OperationPass<ModuleOp>> createLoSPNBufferizePass();

      std::unique_ptr<OperationPass<SPNKernel>> createLoSPNCopyRemovalPass();

      std::unique_ptr<OperationPass<SPNKernel>> createLoSPNPartitionerPass(int maxTaskSize = -1);

      std::unique_ptr<OperationPass<ModuleOp>> createReplaceARMOptimizedRoutinesPass();

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
