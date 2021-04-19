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
    /// Read & parse query from binary input file and execute the compiler.
    /// \param inputFile Path of the input file.
    /// \param options Configuration of the compiler execution.
    /// \return Generated Kernel.
    static Kernel compileQuery(const std::string& inputFile, const options_t& options);

    ///
    /// Query the compiler for information about supported targets.
    /// \param target Name of the target.
    /// \return true if the compilation for the target is supported,
    ///         false otherwise.
    static bool isTargetSupported(const std::string& target);

    ///
    /// Query the compiler for information about supported feature.
    /// \param feature Name of the feature.
    /// \return true if the compilation with the feature is supported,
    ///         false otherwise.
    static bool isFeatureSupported(const std::string& feature);

  };
}

#endif //SPNC_SPNC_H
