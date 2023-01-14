#pragma once

#include "pipeline/steps/mlir/MLIRPassPipeline.h"


namespace spnc {

class LoSPNtoFPGAConversion : public MLIRPassPipeline<LoSPNtoFPGAConversion> {
public:
  using MLIRPassPipeline<LoSPNtoFPGAConversion>::MLIRPassPipeline;

  void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

  STEP_NAME("lospn-to-fpga")
};

}