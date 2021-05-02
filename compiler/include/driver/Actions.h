//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_ACTIONS_H
#define SPNC_ACTIONS_H

#include <driver/Options.h>

namespace spnc {

  ///
  /// Base class of all actions. Toolchains generate a Job, which in turn is composed from
  /// Actions. Actions can depend on other actions and form a DAG as dependency graph.
  class ActionBase {
  public:
    virtual ~ActionBase() = default;

    void setConfiguration(std::shared_ptr<interface::Configuration> configuration) {
      config = std::move(configuration);
    }

  protected:

    std::shared_ptr<interface::Configuration> config;
  };

  ///
  /// An Action producing a single output.
  /// \tparam Output The type of the output produced by this action.
  template<typename Output>
  class ActionWithOutput : public ActionBase {
  public:
    /// Execute the action.
    /// \return A reference to the product of this action.
    virtual Output& execute() = 0;

    ///
    /// Default constructor.
    explicit ActionWithOutput() = default;

    ///
    /// Default destructor.
    ~ActionWithOutput() override = default;
  };

  /// An action depending on a single action and producing a single output.
  /// \tparam Input The type of the product produced by the previous action.
  /// \tparam Output The type of the output produced by this action.
  template<typename Input, typename Output>
  class ActionSingleInput : public ActionWithOutput<Output> {

  public:
    /// Constructor.
    /// \param _input Reference to the action on which this action depends.
    explicit ActionSingleInput(ActionWithOutput<Input>& _input) : input{_input} {}

  protected:
    ///
    /// Reference to action providing input.
    ActionWithOutput<Input>& input;

  };

  /// An action depending on two actions and producing a single output.
  /// \tparam Input1 The type of the product produced by the first previous action.
  /// \tparam Input2 The type of the product produced by the second previous action.
  /// \tparam Output The type of the output produced by this action.
  template<typename Input1, typename Input2, typename Output>
  class ActionDualInput : public ActionWithOutput<Output> {

  public:
    /// Constructor.
    /// \param _input1 Reference to the first action on which this action depends.
    /// \param _input2 Reference to the second action on which this action depends.
    ActionDualInput(ActionWithOutput<Input1>& _input1, ActionWithOutput<Input2>& _input2)
        : input1{_input1}, input2{_input2} {}

  protected:

    ///
    /// Reference to action providing first input.
    ActionWithOutput<Input1>& input1;

    ///
    /// Reference to action providing second input.
    ActionWithOutput<Input2>& input2;

  };
}

#endif //SPNC_ACTIONS_H
