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

    public:
      HiSPNtoLoSPNNodeConversionPass(bool useLogSpaceComputation, bool useOptimalRepresentation) :
          computeLogSpace{useLogSpaceComputation}, optimizeRepresentation{useOptimalRepresentation} {}

    protected:

      void runOnOperation() override;

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

    private:

      bool computeLogSpace;
      bool optimizeRepresentation;

    };

    std::unique_ptr<Pass> createHiSPNtoLoSPNQueryConversionPass(bool useLogSpaceComputation,
                                                                bool useOptimalRepresentation);

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H
