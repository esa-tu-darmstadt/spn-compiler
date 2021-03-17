//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace spn {

    struct HiSPNtoLoSPNNodeConversionPass : public PassWrapper<HiSPNtoLoSPNNodeConversionPass,
                                                               OperationPass < ModuleOp>> {

    public:
    explicit HiSPNtoLoSPNNodeConversionPass(bool useLogSpaceComputation) : computeLogSpace{useLogSpaceComputation} {}

    protected:
    void runOnOperation()
    override;

    private:
    bool computeLogSpace;

  };

  std::unique_ptr<Pass> createHiSPNtoLoSPNNodeConversionPass(bool useLogSpaceComputation);

  struct HiSPNtoLoSPNQueryConversionPass : public PassWrapper<HiSPNtoLoSPNQueryConversionPass,
                                                              OperationPass < ModuleOp>> {

  public:
  explicit HiSPNtoLoSPNQueryConversionPass(bool useLogSpaceComputation) : computeLogSpace{useLogSpaceComputation} {}

  protected:
  void runOnOperation()
  override;

  private:
  bool computeLogSpace;

};

std::unique_ptr<Pass> createHiSPNtoLoSPNQueryConversionPass(bool useLogSpaceComputation);

}
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_HISPNTOLOSPNCONVERSIONPASSES_H
