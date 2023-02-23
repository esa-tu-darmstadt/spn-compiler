//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "../include/spnc-runtime.h"
#include <util/Logging.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/cfg/env.h>

#ifdef TAPASCO_FPGA_SUPPORT
#include "../wrappers/tapasco/TapascoWrapper.hpp"
#endif

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

void spn_runtime::execute(const Kernel& kernel, size_t num_elements, void* inputs, void* outputs) {
  if (kernel.getKernelType() == KernelType::CLASSICAL_KERNEL) {
    ClassicalKernel classicalKernel = kernel.getClassicalKernel();

    // Caching executables wrapping around kernels to avoid repeated loading via libelf.
    if (!cached_executables.count(classicalKernel.uniqueId())) {
      cached_executables.emplace(std::pair<size_t, std::unique_ptr<Executable>>{classicalKernel.uniqueId(),
                                                                                std::make_unique<Executable>(classicalKernel)});
    }
    cached_executables[classicalKernel.uniqueId()]->execute(num_elements, inputs, outputs);
  } else if (kernel.getKernelType() == KernelType::FPGA_KERNEL) {
#ifdef TAPASCO_FPGA_SUPPORT
    initTapasco(kernel)->executeQuery(num_elements, inputs, outputs);
#else
    throw std::runtime_error("TAPASCO_FPGA_SUPPORT is not defined and therefore not implemented");
#endif
  }
}
