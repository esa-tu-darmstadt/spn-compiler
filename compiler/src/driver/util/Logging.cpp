//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/Logging.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/cfg/env.h>

namespace spnc {

  static bool initLogger() {
    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::cfg::load_env_levels();
    spdlog::set_pattern("[%c] [%s:%#] -- [%l] %v");
    return true;
  }

  static bool initOnce = initLogger();

}


