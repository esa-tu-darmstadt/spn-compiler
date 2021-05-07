//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    struct LoSPNtoCPUStructureConversionPass :
        public PassWrapper<LoSPNtoCPUStructureConversionPass, OperationPass<ModuleOp>> {

    public:

      explicit LoSPNtoCPUStructureConversionPass(bool enableVectorization) : vectorize{enableVectorization} {}

    protected:
      void runOnOperation() override;

    public:
      void getDependentDialects(DialectRegistry& registry) const override;

    private:

      bool vectorize;

    };

    std::unique_ptr<Pass> createLoSPNtoCPUStructureConversionPass(bool enableVectorization);

    struct LoSPNtoCPUNodeConversionPass :
        public PassWrapper<LoSPNtoCPUNodeConversionPass, OperationPass<ModuleOp>> {

    protected:
      void runOnOperation() override;

    public:
      void getDependentDialects(DialectRegistry& registry) const override;

    };

    std::unique_ptr<Pass> createLoSPNtoCPUNodeConversionPass();

    struct LoSPNNodeVectorizationPass : public PassWrapper<LoSPNNodeVectorizationPass, OperationPass<ModuleOp>> {

    protected:
      void runOnOperation() override;

    public:
      void getDependentDialects(DialectRegistry& registry) const override;
    };

    std::unique_ptr<Pass> createLoSPNNodeVectorizationPass();

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H
