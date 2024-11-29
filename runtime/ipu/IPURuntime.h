//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

#include <Kernel.h>

namespace poplar {
class Engine;
class Target;
class Device;
} // namespace poplar

namespace spnc_rt {
namespace ipu {
class IPURuntime {
public:
  IPURuntime();
  ~IPURuntime();

  void unload();

  void load(spnc::Kernel &kernel);
  void execute(size_t num_elements, void *inputs, void *outputs);

  spnc::IPUKernel &getLoadedKernel() { return *loadedKernel; }

private:
  bool attach(spnc::IPUTarget targetArch, unsigned numTiles);

  spnc::IPUKernel *loadedKernel = nullptr;

  std::unique_ptr<::poplar::Engine> engine;
  std::unique_ptr<::poplar::Target> target;
  std::unique_ptr<::poplar::Device> device;
};
} // namespace ipu
} // namespace spnc_rt