//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H

#include "mlir/IR/Operation.h"
#include "LoSPN/LoSPNOps.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        bool areConsecutiveLoads(low::SPNBatchRead load1, low::SPNBatchRead load2);
        bool areConsecutiveLoads(std::vector<Operation*> const& loads);
        bool canBeGathered(std::vector<Operation*> const& loads);
        bool areBroadcastable(std::vector<Operation*> const& ops);

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H
