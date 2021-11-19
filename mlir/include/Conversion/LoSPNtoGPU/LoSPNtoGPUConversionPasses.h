//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUCONVERSIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUCONVERSIONPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    struct LoSPNtoGPUStructureConversionPass :
        public PassWrapper<LoSPNtoGPUStructureConversionPass, OperationPass<ModuleOp>> {

    protected:
      void runOnOperation() override;

    public:
      void getDependentDialects(DialectRegistry& registry) const override;
    };

    std::unique_ptr<Pass> createLoSPNtoGPUStructureConversionPass();

    struct LoSPNtoGPUNodeConversionPass :
        public PassWrapper<LoSPNtoGPUNodeConversionPass, OperationPass<ModuleOp>> {

    protected:
      void runOnOperation() override;

    public:
      void getDependentDialects(DialectRegistry& registry) const override;

    };

    std::unique_ptr<Pass> createLoSPNtoGPUNodeConversionPass();

    struct LoSPNGPUSharedMemoryInsertionPass :
        public PassWrapper<LoSPNGPUSharedMemoryInsertionPass, OperationPass<ModuleOp>> {

    protected:
      void runOnOperation() override;

    };

    std::unique_ptr<Pass> createLoSPNGPUSharedMemoryInsertionPass();

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOGPU_LOSPNTOGPUCONVERSIONPASSES_H
