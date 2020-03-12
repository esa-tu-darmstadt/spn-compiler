//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "../include/spnc-runtime.h"
#include <util/Logging.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/cfg/env.h>

namespace spnc_rt {

  static bool initLogger() {
    /// Statically initialize the logger to log to the terminal output.
    if (!spdlog::get("console").get()) {
      auto console = spdlog::stdout_color_mt("console");
      spdlog::set_default_logger(console);
      spdlog::cfg::load_env_levels();
      spdlog::set_pattern("[%c] [%s:%#] -- [%l] %v");
    }
    return true;
  }

  static bool initOnce = initLogger();

}

using namespace spnc_rt;

spn_runtime* spn_runtime::_instance = nullptr;

spn_runtime& spn_runtime::instance() {
  if (!_instance) {
    _instance = new spn_runtime{};
  }
  return *_instance;
}

void spn_runtime::execute(const Kernel& kernel, size_t num_elements, void* inputs, double* outputs) {
  // Caching executables wrapping around kernels to avoid repeated loading via libelf.
  if (!cached_executables.count(kernel.unique_id())) {
    cached_executables.emplace(std::pair<size_t, std::unique_ptr<Executable>>{kernel.unique_id(),
                                                                              std::make_unique<Executable>(kernel)});
  }
  cached_executables[kernel.unique_id()]->execute(num_elements, inputs, outputs);
}
