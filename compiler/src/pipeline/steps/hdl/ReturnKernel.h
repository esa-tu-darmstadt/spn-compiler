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

  ExecutionResult execute() override { return success(); }

  Kernel *result() override { return getContext()->get<Kernel>(); }

  STEP_NAME("return-kernel")
};

}