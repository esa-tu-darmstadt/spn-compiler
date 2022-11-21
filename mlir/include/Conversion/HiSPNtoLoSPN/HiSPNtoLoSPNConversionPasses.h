//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
  namespace spn {

    struct HiSPNtoLoSPNNodeConversionPass :
        public PassWrapper<HiSPNtoLoSPNNodeConversionPass, OperationPass<ModuleOp>> {

    public:
      HiSPNtoLoSPNNodeConversionPass(bool useLogSpaceComputation, bool useOptimalRepresentation) :
          computeLogSpace{useLogSpaceComputation}, optimizeRepresentation{useOptimalRepresentation} {}

    protected:

      void runOnOperation() override;

    public:
      void getDependentDialects(DialectRegistry& registry) const override;

      // TODO: This somehow identifies the passes. How does it work exactly?
      StringRef getArgument() const override { return "convert-hispn-node-to-lospn"; }
      StringRef getDescription() const override { return "Convert nodes from HiSPN to LoSPN dialect"; }
    private:

      bool computeLogSpace;
      bool optimizeRepresentation;

    };

    std::unique_ptr<Pass> createHiSPNtoLoSPNNodeConversionPass(bool useLogSpaceComputation,
                                                               bool useOptimalRepresentation);

    struct HiSPNtoLoSPNQueryConversionPass :
        public PassWrapper<HiSPNtoLoSPNQueryConversionPass, OperationPass<ModuleOp>> {

    public:
      HiSPNtoLoSPNQueryConversionPass(bool useLogSpaceComputation, bool useOptimalRepresentation) :
          computeLogSpace{useLogSpaceComputation}, optimizeRepresentation{useOptimalRepresentation} {}

    protected:

      void runOnOperation() override;

    public:
      void getDependentDialects(DialectRegistry& registry) const override;

      StringRef getArgument() const override { return "convert-hispn-query-to-lospn"; }
      StringRef getDescription() const override { return "Convert queries from HiSPN to LoSPN dialect"; }
    private:

      bool computeLogSpace;
      bool optimizeRepresentation;

    };

    std::unique_ptr<Pass> createHiSPNtoLoSPNQueryConversionPass(bool useLogSpaceComputation,
                                                                bool useOptimalRepresentation);

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H
