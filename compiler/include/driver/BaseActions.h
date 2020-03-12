//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_BASEACTIONS_H
#define SPNC_BASEACTIONS_H

#include <fstream>
#include <memory>
#include "Actions.h"
#include <util/FileSystem.h>

namespace spnc {

  ///
  /// Action for making the contents of a file available as
  /// string to depending actions.
  class FileInputAction : public ActionWithOutput<std::string> {
  public:
    /// Constructor.
    /// \param _fileName Name of the file to read.
    explicit FileInputAction(const std::string& _fileName) : fileName{_fileName} {}

    /// Read the file contents into a string.
    /// \return A string containing the file's content.
    std::string& execute() override {
      if (!cached) {
        content = std::make_unique<std::string>();
        std::ifstream in(fileName);
        if (!in) {
          std::cerr << "ERROR: Could not read file " << fileName << std::endl;
          throw std::system_error{};
        }
        in.seekg(0, std::ios::end);
        content->resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(&(*content)[0], content->size());
        in.close();
      }
      return *content;
    }

  private:
    std::string fileName;
    std::unique_ptr<std::string> content;
    bool cached = false;

  };

  ///
  /// Action for making the contents of a string available to depending actions.
  class StringInputAction : public ActionWithOutput<std::string> {
  public:

    /// Constructor.
    /// \param s The string content.
    explicit StringInputAction(const std::string& s) : content{std::make_unique<std::string>(s)} {}

    std::string& execute() override { return *content; }

  private:
    std::unique_ptr<std::string> content;
  };

  /// A JoinAction allows to merge divergent branches of the action dependency graph.
  /// The output of this action corresponds to the output of the first input action.
  /// \tparam InOut Output type of this action and the first input action.
  /// \tparam SecondIn Output type of the second input action.
  template<typename InOut, typename SecondIn>
  class JoinAction : public ActionDualInput<InOut, SecondIn, InOut> {

  public:
    /// Constructor.
    /// \param _input1 The first input action, its result will be passed on.
    /// \param _input2 The second input action.
    JoinAction(ActionWithOutput<InOut>& _input1, ActionWithOutput<SecondIn>& _input2)
        : ActionDualInput<InOut, SecondIn, InOut>{_input1, _input2} {}

    /// Trigger the execution of both input actions and pass on the result of the first input action.
    /// \return The result of the first input action.
    InOut& execute() override {
      this->input2.execute();
      return this->input1.execute();
    }

  };
}

#endif //SPNC_BASEACTIONS_H
