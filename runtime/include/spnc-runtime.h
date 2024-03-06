//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_RUNTIME_H
#define SPNC_RUNTIME_H

#include "../src/Executable.h"
#include <Kernel.h>
#include <memory>
#include <string>
#include <unordered_map>

using namespace spnc;

namespace spnc_rt {

///
/// Entry point of the runtime.
class spn_runtime {

public:
  /// Get the currently active instance of the runtime (Singleton pattern).
  /// \return Reference to the currently active runtime instance.
  static spn_runtime &instance();

  /// Launch the kernel with the given inputs.
  /// \param kernel Kernel to launch.
  /// \param num_elements Number of queries in the batch.
  /// \param inputs Input SPN evidence.
  /// \param outputs Results computed by the kernel.
  void execute(const Kernel &kernel, size_t num_elements, void *inputs,
               void *outputs);

  spn_runtime(const spn_runtime &) = delete;

  spn_runtime(spn_runtime &&) = delete;

  spn_runtime &operator=(const spn_runtime &) = delete;

  spn_runtime &operator=(spn_runtime &&) = delete;

private:
  explicit spn_runtime() = default;

  static spn_runtime *_instance;

  std::unordered_map<size_t, std::unique_ptr<Executable>> cached_executables;
};

} // namespace spnc_rt

#endif // SPNC_RUNTIME_H
