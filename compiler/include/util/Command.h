//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMMAND_H
#define SPNC_COMMAND_H

#include <vector>
#include <sstream>
#include <iostream>
#include "Logging.h"

namespace spnc {

  ///
  /// Utility class providing facilities to invoke external commands.
  class Command {

  public:

    /// Execute an external command.
    /// \param command Command string.
    /// \return Return code of the external command.
    static int executeExternalCommand(const std::string& command) {
      if (!system(nullptr)) {
        SPNC_FATAL_ERROR("No processor for external commands available!");
      }
      return system(command.c_str());
    }

    /// Compose and execute an external command. Throw an error if the external command did not
    /// completely successfully.
    /// \param command Components of the command.
    /// \return The return code of the external command.
    static void executeExternalCommand(const std::vector<std::string>& command) {
      std::ostringstream oss;
      for (auto& c : command) {
        oss << c << " ";
      }
      int ret = executeExternalCommand(oss.str());
      if (ret) {
        SPNC_FATAL_ERROR("Error executing external command {}, return code: {}", oss.str(), ret);
      }
    }

  private:
    explicit Command() = default;

  };

}

#endif //SPNC_COMMAND_H
