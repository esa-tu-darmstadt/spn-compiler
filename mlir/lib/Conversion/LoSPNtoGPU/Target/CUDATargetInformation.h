//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_CONVERSION_LOSPNTOGPU_TARGET_CUDATARGETINFORMATION_H
#define SPNC_MLIR_LIB_CONVERSION_LOSPNTOGPU_TARGET_CUDATARGETINFORMATION_H

#include "mlir/IR/Diagnostics.h"

namespace mlir {
  namespace spn {

    class CUDATargetInformation {

    public:

      static unsigned maxSharedMemoryPerBlock(mlir::Location loc);

    };

  }
}

#endif //SPNC_MLIR_LIB_CONVERSION_LOSPNTOGPU_TARGET_CUDATARGETINFORMATION_H
