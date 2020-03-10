//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H
#define SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>

namespace spnc {
  namespace logging {
    static bool initLogger() {
      auto console = spdlog::stdout_color_mt("console");
      spdlog::set_default_logger(console);
      spdlog::cfg::load_env_levels();
      spdlog::set_pattern("[%c] [%s:%#] -- [%l] %v");
      return true;
    }

    static bool initOnce = initLogger();
  }
}

#endif //SPNC_COMPILER_INCLUDE_UTIL_LOGGING_H
