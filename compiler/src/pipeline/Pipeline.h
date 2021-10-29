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

    extern interface::Option<std::string> stopAfter;

  }

  template<typename Result>
  class Pipeline : public PipelineBase {

  public:

    using PipelineBase::PipelineBase;

    ExecutionResult execute() {
      auto* config = context->template get<interface::Configuration>();
      llvm::Optional<std::string> stop;
      if (option::stopAfter.isPresent(*config)) {
        stop = option::stopAfter.get(*config);
      }
      for (auto& step: steps) {
        // TODO Stop after/before
        auto result = step->execute();
        if (failed(result)) {
          return result;
        }
        if (stop.hasValue() && step->name() == stop.getValue()) {
          return failure("STOPPED PIPELINE after {}", stop.getValue());
        }
      }
      if (stop.hasValue()) {
        SPDLOG_WARN("Did not stop after {}, because no such step was present in the pipeline", stop.getValue());
      }
      if (!valid) {
        return failure("INVALID PIPELINE");
      }
      return success();
    }

    Result* result() {
      assert(valid && lastStep && "Cannot get result from invalid pipeline");
      return lastStep->result();
    }

    void toText() {
      std::cout << "Pipeline: " << std::endl;
      for (auto& s: steps) {
        std::cout << "\t" << s->name() << std::endl;
      }
    }

    template<class Step,
        typename std::enable_if<std::is_base_of<StepWithResult < Result>, Step>::value, Step> ::type* = nullptr,
    typename ... Args>
    Step& emplaceStep(Args&& ... args) {
      auto step = emplace<Step>(std::forward<Args>(args)...);
      valid = true;
      lastStep = step;
      return *step;
    }

    template<class Step,
        typename std::enable_if<!std::is_base_of<StepWithResult < Result>, Step>::value, Step> ::type* = nullptr,
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

    StepWithResult <Result>* lastStep = nullptr;

  };

}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINE_H
