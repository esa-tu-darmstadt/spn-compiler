//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_RUNTIME_H
#define SPNC_RUNTIME_H

#include <string>
#include <Kernel.h>
#include <unordered_map>
#include <memory>
#include "../src/Executable.h"

using namespace spnc;

namespace spnc_rt {

  ///
  /// Entry point of the runtime.
  class spn_runtime {

  public:

    /// Get the currently active instance of the runtime (Singleton pattern).
    /// \return Reference to the currently active runtime instance.
    static spn_runtime& instance();

    /// Launch the kernel with the given inputs.
    /// \param kernel Kernel to launch.
    /// \param num_elements Number of queries in the batch.
    /// \param inputs Input SPN evidence.
    /// \param outputs Probabilities computed by the SPN.
    void execute(const Kernel& kernel, size_t num_elements, void* inputs, double* outputs);

    spn_runtime(const spn_runtime&) = delete;

    spn_runtime(spn_runtime&&) = delete;

    spn_runtime& operator=(const spn_runtime&) = delete;

    spn_runtime& operator=(spn_runtime&&) = delete;

  private:

    explicit spn_runtime() = default;

    static spn_runtime* _instance;

    std::unordered_map<size_t, std::unique_ptr<Executable>> cached_executables;

  };

}

#endif //SPNC_RUNTIME_H
