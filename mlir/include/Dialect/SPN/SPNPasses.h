//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_SPN_SPNPASSES_H
#define SPNC_MLIR_DIALECTS_INCLUDE_SPN_SPNPASSES_H

#include "mlir/Pass/Pass.h"
#include "SPNOps.h"

namespace mlir {
  namespace spn {

    /// Instantiate the SPNOpSimplifierPass simplifying operations of
    /// the SPN dialect.
    /// \return Pass instance.
    std::unique_ptr<OperationPass<ModuleOp>> createSPNOpSimplifierPass();

    /// Instantiate the SPNTypePinning pass determining which datatype will be used
    /// to compute the SPN.
    /// \return Pass instance.
    std::unique_ptr<OperationPass<ModuleOp>> createSPNTypePinningPass();

    /// Instantiate the SPNVectorization pass vectorizing the computation of the SPN.
    /// \return Pass instance.
    std::unique_ptr<OperationPass<JointQuery>> createSPNVectorizationPass();

#define GEN_PASS_REGISTRATION
#include "SPN/SPNPasses.h.inc"
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_SPN_SPNPASSES_H
