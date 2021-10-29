//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <util/Logging.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/cfg/env.h>

namespace spnc {

  static bool initLogger() {
    /// Statically initialize the logger to log to the terminal output.
    if (!spdlog::get("console")) {
      auto console = spdlog::stdout_color_mt("console");
      spdlog::set_default_logger(console);
      spdlog::cfg::load_env_levels();
      spdlog::set_pattern("[%c] [%s:%#] -- [%l] %v");
    }
    return true;
  }

  static bool initOnce = initLogger();

}


