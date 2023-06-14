#pragma once

#include "pipeline/steps/mlir/MLIRPassPipeline.h"


namespace spnc {

class LoSPNtoFPGAConversion : public MLIRPassPipeline<LoSPNtoFPGAConversion> {
public:
  LoSPNtoFPGAConversion(const std::string& fpgaConfigJson, StepWithResult<mlir::ModuleOp>& input):
    MLIRPassPipeline<LoSPNtoFPGAConversion>(input), fpgaConfigJson(fpgaConfigJson) {}

  void preProcess(mlir::ModuleOp *inputModule) override;
  void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);
  void postProcess(mlir::ModuleOp *resultModule) override;

  STEP_NAME("lospn-to-fpga")
private:
  std::string fpgaConfigJson;
};

}