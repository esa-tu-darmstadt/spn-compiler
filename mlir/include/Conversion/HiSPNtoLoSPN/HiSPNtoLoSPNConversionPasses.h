//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    struct HiSPNtoLoSPNNodeConversionPass :
        public PassWrapper<HiSPNtoLoSPNNodeConversionPass, OperationPass<ModuleOp>> {

    protected:

      void runOnOperation() override;

    };

    std::unique_ptr<Pass> createHiSPNtoLoSPNNodeConversionPass();

    struct HiSPNtoLoSPNQueryConversionPass :
        public PassWrapper<HiSPNtoLoSPNQueryConversionPass, OperationPass<ModuleOp>> {

    protected:

      void runOnOperation() override;

    };

    std::unique_ptr<Pass> createHiSPNtoLoSPNQueryConversionPass();

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H
