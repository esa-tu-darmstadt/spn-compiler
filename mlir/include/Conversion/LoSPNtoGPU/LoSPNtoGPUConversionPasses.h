//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUCONVERSIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUCONVERSIONPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    struct LoSPNtoGPUStructureConversionPass :
        public PassWrapper<LoSPNtoGPUStructureConversionPass, OperationPass<ModuleOp>> {

    protected:
      void runOnOperation() override;

    };

    std::unique_ptr<Pass> createLoSPNtoGPUStructureConversionPass();

    struct LoSPNtoGPUNodeConversionPass :
        public PassWrapper<LoSPNtoGPUNodeConversionPass, OperationPass<ModuleOp>> {

    protected:
      void runOnOperation() override;

    };

    std::unique_ptr<Pass> createLoSPNtoGPUNodeConversionPass();

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUCONVERSIONPASSES_H
