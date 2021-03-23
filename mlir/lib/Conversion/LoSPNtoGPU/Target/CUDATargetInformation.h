//
// This file is part of the SPNC project.
// Copyright (c) 2021 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

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
