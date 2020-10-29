//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOLLVM_SPNTOLLVMCONVERSIONPASS_H
#define SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOLLVM_SPNTOLLVMCONVERSIONPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    struct SPNtoLLVMConversionPass : public PassWrapper<SPNtoLLVMConversionPass, OperationPass<ModuleOp>> {
    protected:
      void runOnOperation() override;
    };

    std::unique_ptr<Pass> createSPNtoLLVMConversionPass();

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOLLVM_SPNTOLLVMCONVERSIONPASS_H
