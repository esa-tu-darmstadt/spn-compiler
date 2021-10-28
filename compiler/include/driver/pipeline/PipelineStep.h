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

#define STEP_NAME(NAME) inline static std::string stepName(){ return NAME; }

namespace spnc {

  class ExecutionResult;

  ExecutionResult failure(std::string message);
  ExecutionResult success();

  class ExecutionResult final {

  public:

    bool successful() const {
      return success;
    };

    std::string message() const {
      return msg;
    }

  private:

    ExecutionResult() : success{true}, msg{"SUCCESS"} {}

    explicit ExecutionResult(std::string message) : success{false}, msg{std::move(message)} {}

    bool success;

    std::string msg;

    friend ExecutionResult spnc::failure(std::string message);
    friend ExecutionResult spnc::success();

  };

  template<typename... Args>
  ExecutionResult failure(std::string message, Args&& ... args) {
    // Use the fmt library for formatting, as it comes with spdlog anyways.
    // Could be replaced by std::format after the move to C++20.
    return failure(fmt::format(message, std::forward<Args>(args)...));
  }

  static inline bool failed(ExecutionResult& result) {
    return !result.successful();
  }

  class StepBase;

  class PipelineBase {

  public:

    PipelineBase() {
      context = std::make_unique<PipelineContext>();
    }

    PipelineContext* getContext();

  protected:

    void setPipeline(StepBase& sb);

    std::unique_ptr<PipelineContext> context;

  };

  class StepBase {

  public:

    explicit StepBase(std::string stepName) : pipeline{nullptr}, _name(std::move(stepName)) {}

    virtual ~StepBase() = default;

    // TODO: Execute only once?
    virtual ExecutionResult execute() = 0;

    std::string name() {
      return _name;
    }

  protected:

    PipelineContext* getContext() {
      assert(pipeline);
      return pipeline->getContext();
    }

  private:

    friend PipelineBase;

    PipelineBase* pipeline;

    std::string _name;

  };

  template<class Result>
  class StepWithResult {

  public:

    virtual Result* result() = 0;

  };

  template<class Step, class Input>
  class StepSingleInput : public StepBase {

  public:

    explicit StepSingleInput(StepWithResult<Input>& input) : StepBase(std::move(Step::stepName())), in{input} {}

    ExecutionResult execute() override {
      return static_cast<Step&>(*this).executeStep(in.result());
    }

  protected:

    StepWithResult<Input>& in;
  };

  template<class Step, class Input1, class Input2>
  class StepDualInput : public StepBase {

  public:

    StepDualInput(StepWithResult<Input1>& input1, StepWithResult<Input2>& input2)
        : StepBase(std::move(Step::stepName())),
          in1{input1}, in2{input2} {}

    ExecutionResult execute() override {
      return static_cast<Step&>(*this).executeStep(in1.result(), in2.result());
    }

  protected:

    StepWithResult<Input1>& in1;

    StepWithResult<Input2>& in2;

  };

}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_PIPELINESTEP_H
