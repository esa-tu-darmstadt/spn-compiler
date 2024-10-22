//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_PIPELINESTEP_H
#define SPNC_COMPILER_INCLUDE_DRIVER_PIPELINESTEP_H

#include "PipelineContext.h"
#include "spdlog/fmt/fmt.h"

#define STEP_NAME(NAME)                                                                                                \
  inline static std::string stepName() { return NAME; }

namespace spnc {

class ExecutionResult;

/// Indicate a failure.
/// \param message Debug message.
/// \return Failed ExecutionResult with the given message.
ExecutionResult failure(std::string message);

/// Indidate a success.
/// \return Successful ExecutionResult.
ExecutionResult success();

///
/// Value to indicate success or failure of a step.
class ExecutionResult final {

public:
  /// Check for success.
  /// \return true if the underlying result was successful, false otherwise.
  bool successful() const { return success; };

  /// Get the message.
  /// \return Message.
  std::string message() const { return msg; }

private:
  ExecutionResult() : success{true}, msg{"SUCCESS"} {}

  explicit ExecutionResult(std::string message) : success{false}, msg{std::move(message)} {}

  bool success;

  std::string msg;

  friend ExecutionResult spnc::failure(std::string message);
  friend ExecutionResult spnc::success();
};

/// Indicate a failure.
/// \tparam Args Types of the format string arguments
/// \param message Format string.
/// \param args Arguments to the format string.
/// \return Failed ExecutionResult with the given, formatted message.
template <typename... Args> ExecutionResult failure(std::string message, Args &&...args) {
  // Use the fmt library for formatting, as it comes with spdlog anyways.
  // Could be replaced by std::format after the move to C++20.
  return failure(fmt::format(message, std::forward<Args>(args)...));
}

static inline bool failed(ExecutionResult &result) { return !result.successful(); }

///
/// Base class for all steps.
class StepBase;

///
/// Base class for compilation pipelines.
class PipelineBase {

public:
  PipelineBase() { context = std::make_unique<PipelineContext>(); }

  /// Get the associated PipelineContext
  /// \return Non-owning pointer to the associated PipelineContext.
  PipelineContext *getContext();

protected:
  void setPipeline(StepBase &sb);

  std::unique_ptr<PipelineContext> context;
};

///
/// Base class for all executable steps.
class StepBase {

public:
  /// Constructor.
  /// \param stepName Name of the step.
  explicit StepBase(std::string stepName) : pipeline{nullptr}, _name(std::move(stepName)) {}

  virtual ~StepBase() = default;

  /// Execute the step.
  /// \return success() if the step executed successfully, failure() otherwise.
  virtual ExecutionResult execute() = 0;

  /// Get the name.
  /// \return Name of the step.
  std::string name() { return _name; }

protected:
  PipelineContext *getContext() {
    assert(pipeline);
    return pipeline->getContext();
  }

private:
  friend PipelineBase;

  PipelineBase *pipeline;

  std::string _name;
};

/// Mixin for all steps producing a result usable by other steps.
/// \tparam Result Type of the result.
template <class Result> class StepWithResult {

public:
  /// Retrieve the result.
  /// \return Non-owning pointer to the result.
  virtual Result *result() = 0;
};

/// CRTP base for steps consuming a single input.
/// \tparam Step CRTP parameter.
/// \tparam Input Type of the input.
template <class Step, class Input> class StepSingleInput : public StepBase {

public:
  /// Constructor.
  /// Subclasses must implement a static class-method 'std::string stepName()'.
  /// \param input Reference to the step producing the input.
  explicit StepSingleInput(StepWithResult<Input> &input) : StepBase(std::move(Step::stepName())), in{input} {}

  /// CRTP method, sub-classes must implement a method 'ExecutionResult
  /// executeStep(Input*). \return success() if the step executed successfully,
  /// failure() otherwise.
  ExecutionResult execute() override { return static_cast<Step &>(*this).executeStep(in.result()); }

protected:
  StepWithResult<Input> &in;
};

/// CRTP base for steps consuming two inputs.
/// \tparam Step CRTP parameter.
/// \tparam Input1 Type of the first input.
/// \tparam Input2 Type of the second input.
template <class Step, class Input1, class Input2> class StepDualInput : public StepBase {

public:
  /// Constructor.
  /// Subclasses must implement a static class-method 'std::string stepName()'.
  /// \param input1 Reference to the step producing the first input.
  /// \param input2 Refernece to the step producing the second input.
  StepDualInput(StepWithResult<Input1> &input1, StepWithResult<Input2> &input2)
      : StepBase(std::move(Step::stepName())), in1{input1}, in2{input2} {}

  /// CRTP method, sub-classes must implement a method 'ExecutionResult
  /// executeStep(Input1*, Input2*). \return success() if the step executed
  /// successfully, failure() otherwise.
  ExecutionResult execute() override { return static_cast<Step &>(*this).executeStep(in1.result(), in2.result()); }

protected:
  StepWithResult<Input1> &in1;

  StepWithResult<Input2> &in2;
};

} // namespace spnc

#endif // SPNC_COMPILER_INCLUDE_DRIVER_PIPELINESTEP_H
