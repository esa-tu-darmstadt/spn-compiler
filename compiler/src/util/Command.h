//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMMAND_H
#define SPNC_COMMAND_H

#include <vector>
#include <sstream>
#include <iostream>
#include "util/Logging.h"

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

    static std::string executeExternalCommandAndGetOutput(const std::string& cmd) {
      // Taken from
      // https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
      std::array<char, 128> buffer;
      std::string result;
      std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
      if (!pipe) {
          throw std::runtime_error("popen() failed!");
      }
      while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
          result += buffer.data();
          printf("%s", buffer.data());
      }
      return result;
    }

  private:
    explicit Command() = default;

  };

}

#endif //SPNC_COMMAND_H
