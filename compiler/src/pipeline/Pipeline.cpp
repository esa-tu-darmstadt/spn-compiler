//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <pipeline/Pipeline.h>
#include <pipeline/PipelineStep.h>

unsigned spnc::PipelineContext::lastTypeID = 0;

void spnc::PipelineBase::setPipeline(StepBase &sb) { sb.pipeline = this; }

spnc::PipelineContext *spnc::PipelineBase::getContext() {
  return context.get();
}

spnc::ExecutionResult spnc::failure(std::string message) {
  return ExecutionResult(std::move(message));
}

spnc::ExecutionResult spnc::success() { return ExecutionResult{}; }

spnc::interface::Option<std::string> spnc::option::stopAfter {
  "stopAfter", "Stop after the specified pipeline step.", "Compilation"
};