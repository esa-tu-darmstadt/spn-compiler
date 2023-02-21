#pragma once

#include "pipeline/PipelineStep.h"


namespace spnc {

/**
 * This class is just for runtime when the user does not want to perform
 * any SPN compilation but still needs vital information for executing on
 * a FPGA.
*/
class ReturnKernel : public StepBase, public StepWithResult<Kernel> {
public:
  explicit ReturnKernel(): StepBase("return-kernel") {}

  ExecutionResult execute() override {
    setKernel(getContext());
    return success();
  }

  Kernel *result() override { return getContext()->get<Kernel>(); }

  STEP_NAME("return-kernel")

  static void setKernel(PipelineContext *context) {
    KernelInfo *kernelInfo = context->get<KernelInfo>();
    Kernel kernel{FPGAKernel()};
    context->add<Kernel>(std::move(kernel));
    FPGAKernel& fpgaKernel = context->get<Kernel>()->getFPGAKernel();

    fpgaKernel.spnVarCount = kernelInfo->numFeatures;
    fpgaKernel.spnBitsPerVar = kernelInfo->bytesPerFeature * 8; // TODO
    fpgaKernel.spnResultWidth = 64; // double precision float

    fpgaKernel.mAxisControllerWidth = round8(fpgaKernel.spnResultWidth);
    fpgaKernel.sAxisControllerWidth = round8(fpgaKernel.spnBitsPerVar * fpgaKernel.spnVarCount);

    // TODO: Make this parameterizable
    fpgaKernel.memDataWidth = 32;
    fpgaKernel.memAddrWidth = 32;

    fpgaKernel.liteDataWidth = 32;
    fpgaKernel.liteAddrWidth = 32;
  }
};

}