#pragma once

#include "pipeline/steps/mlir/MLIRPassPipeline.h"


namespace spnc {

class LoSPNtoFPGAConversion : public MLIRPassPipeline<LoSPNtoFPGAConversion> {
  uint32_t floatMantissaWidth;
  uint32_t floatExponentWidth;
  bool use32Bit;
public:
  LoSPNtoFPGAConversion(const std::string& fpgaConfigJson, uint32_t floatMantissaWidth, uint32_t floatExponentWidth, bool use32Bit, StepWithResult<mlir::ModuleOp>& input):
    MLIRPassPipeline<LoSPNtoFPGAConversion>(input), fpgaConfigJson(fpgaConfigJson), floatMantissaWidth(floatMantissaWidth), floatExponentWidth(floatExponentWidth), use32Bit(use32Bit) {}

  void preProcess(mlir::ModuleOp *inputModule) override;
  void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);
  void postProcess(mlir::ModuleOp *resultModule) override;

  STEP_NAME("lospn-to-fpga")
private:
  std::string fpgaConfigJson;
};

}