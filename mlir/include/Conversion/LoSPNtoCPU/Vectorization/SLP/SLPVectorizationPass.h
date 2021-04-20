//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H

#include <set>
#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        struct SLPVectorizationPass : public PassWrapper<SLPVectorizationPass, FunctionPass> {

        protected:
          void runOnFunction() override;

        };
      }
    }
    std::unique_ptr<Pass> createSLPVectorizationPass();
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H
