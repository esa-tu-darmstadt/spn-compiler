//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H

#include "mlir/IR/Operation.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        bool areConsecutiveLoads(Operation* load1, Operation* load2);
        bool areConsecutiveLoads(std::vector<Operation*> const& loads);

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPUTIL_H
