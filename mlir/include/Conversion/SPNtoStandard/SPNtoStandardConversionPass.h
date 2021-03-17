//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDCONVERSIONPASS_H
#define SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDCONVERSIONPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    ///
    /// Pass performing (partial)lowering from combination of SPN to Standard dialect.
    struct SPNtoStandardConversionPass : public PassWrapper<SPNtoStandardConversionPass, OperationPass<ModuleOp>> {

    public:

      explicit SPNtoStandardConversionPass(bool cpuVectorize);

    protected:
      void runOnOperation() override;

    private:

      bool vectorize;
    };

    /// Instantiate the SPNtoStandardConversionPass.
    /// \return Pass instance.
    std::unique_ptr<Pass> createSPNtoStandardConversionPass(bool cpuVectorize);

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_CONVERSION_SPNTOSTANDARD_SPNTOSTANDARDCONVERSIONPASS_H
