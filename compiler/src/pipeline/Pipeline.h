//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINE_H
#define SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINE_H

#include "PipelineStep.h"
#include "option/Options.h"
#include <vector>
#include <type_traits>
#include <iostream>

namespace spnc {

  namespace option {

    ///
    /// Option to specify a step after which the
    /// compilation pipeline should be terminated prematurely.
    extern interface::Option<std::string> stopAfter;

  }

  /// Representation of a compilation pipeline composed from individual steps.
  /// If valid, i.e., if the last step produces a result of the correct type,
  /// the final result can be retrieved from the pipeline.
  /// Also holds a context (see [PipelineContext]) to pass information across steps.
  /// \tparam Result Type of the final result.
  template<typename Result>
  class Pipeline : public PipelineBase {

  public:

    using PipelineBase::PipelineBase;

    /// Execute the pipeline by executing all steps in the order of insertion.
    /// \return success if all steps executed successfully and the pipeline is valid,
    ///         failure otherwise.
    ExecutionResult execute() {
      auto* config = context->template get<interface::Configuration>();
      llvm::Optional<std::string> stop;
      if (option::stopAfter.isPresent(*config)) {
        stop = option::stopAfter.get(*config);
      }
      for (auto& step: steps) {
        auto result = step->execute();
        if (failed(result)) {
          return result;
        }
        if (stop.hasValue() && step->name() == stop.getValue()) {
          // Stop after the step, if the user requested to do so via the 'stopAfter' option.
          return failure("STOPPED PIPELINE after {}", stop.getValue());
        }
      }
      if (stop.hasValue()) {
        SPDLOG_WARN("Did not stop after {}, because no such step was present in the pipeline", stop.getValue());
      }
      if (!valid) {
        return failure("INVALID PIPELINE");
      }
      executed = true;
      return success();
    }

    /// Retrieve the final result after execution.
    /// \return The final result of the compilation pipeline as produced by the last step.
    Result* result() {
      assert(executed && valid && lastStep && "Cannot get result from not executed or invalid pipeline");
      return lastStep->result();
    }

    std::string toText() {
      static std::string text = [this]() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
      }();
      return text;
    }

    template<class R>
    friend std::ostream& operator<<(std::ostream& os, const Pipeline<R>& p);

    /// Add a step to the pipeline.
    /// \tparam Step Class of the step.
    /// \tparam Args Types of the arguments for the constructor of the step.
    /// \param args Constructor arguments for the constructor of the step.
    /// \return Reference to the emplaced step.
    template<class Step,
        typename std::enable_if<std::is_base_of<StepWithResult<Result>, Step>::value, Step>::type* = nullptr,
        typename ... Args>
    Step& emplaceStep(Args&& ... args) {
      auto step = emplace<Step>(std::forward<Args>(args)...);
      valid = true;
      lastStep = step;
      return *step;
    }

    /// Add a step to the pipeline.
    /// \tparam Step Class of the step.
    /// \tparam Args Types of the arguments for the constructor of the step.
    /// \param args Constructor arguments for the constructor of the step.
    /// \return Reference to the emplaced step.
    template<class Step,
        typename std::enable_if<!std::is_base_of<StepWithResult<Result>, Step>::value, Step>::type* = nullptr,
        typename ... Args>
    Step& emplaceStep(Args&& ... args) {
      auto step = emplace<Step>(std::forward<Args>(args)...);
      valid = false;
      lastStep = nullptr;
      return *step;
    }

  private:

    template<class Step, typename ... Args>
    Step* emplace(Args&& ... args) {
      static_assert(std::is_base_of<StepBase, Step>::value, "Must be a step derived from StepBase");
      auto step = std::make_unique<Step>(std::forward<Args>(args)...);
      steps.push_back(std::move(step));
      auto inserted = steps.back().get();
      setPipeline(*inserted);
      return static_cast<Step*>(inserted);
    }

    std::vector<std::unique_ptr<StepBase>> steps;

    bool valid = false;

    bool executed = false;

    StepWithResult<Result>* lastStep = nullptr;

  };

  template<typename R>
  std::ostream& operator<<(std::ostream& os, const Pipeline<R>& p) {
    os << "Pipeline: " << std::endl;
    for (auto& s: p.steps) {
      os << "\t" << s->name() << std::endl;
    }
    return os;
  }

}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINE_H
