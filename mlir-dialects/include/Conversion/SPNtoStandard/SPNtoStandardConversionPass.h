//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDCONVERSIONPASS_H
#define SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDCONVERSIONPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    struct SPNtoStandardConversionPass : public PassWrapper<SPNtoStandardConversionPass, OperationPass<ModuleOp>> {
    protected:
      void runOnOperation() override;
    };

    std::unique_ptr<Pass> createSPNtoStandardConversionPass();

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDCONVERSIONPASS_H
