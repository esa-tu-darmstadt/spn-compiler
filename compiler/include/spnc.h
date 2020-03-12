//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_SPNC_H
#define SPNC_SPNC_H

#include <string>
#include <Kernel.h>
#include <map>

namespace spnc {

  using options_t = std::map<std::string, std::string>;

  ///
  /// Entry-point of the compiler.
  class spn_compiler {
  public:
    /// Read & parse JSON from input file and execute the compiler.
    /// \param inputFile Path of the input file.
    /// \param options Configuration of the compiler execution.
    /// \return Generated Kernel.
    static Kernel parseJSON(const std::string& inputFile, const options_t& options);

    /// Parse JSON from input string and execute the compiler.
    /// \param jsonString String containing the JSON serialization of the SPN.
    /// \param options Configuration of the compiler execution.
    /// \return Generated Kernel.
    Kernel parseJSONString(const std::string& jsonString, const options_t& options);

  };
}

#endif //SPNC_SPNC_H
