//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "IPURuntime.h"
#include "Kernel.h"
#include "poplar/Target.hpp"
#include "poplar/TargetType.hpp"

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>

#include <util/Logging.h>

using namespace spnc_rt::ipu;
using namespace spnc;

bool IPURuntime::attach(IPUTarget targetArch, unsigned numTiles) {
  if (targetArch == IPUTarget::Model) {
    ::poplar::IPUModel ipuModel;
    device = std::make_unique<poplar::Device>(ipuModel.createDevice());
    spdlog::info("Attached to IPU model");
  } else {
    unsigned numIPUs = 1;
    spdlog::info("Trying to attach to {} IPU(s).", numIPUs);
    poplar::DeviceManager manager =
        poplar::DeviceManager::createDeviceManager();
    std::vector<poplar::Device> devices =
        manager.getDevices(poplar::TargetType::IPU, 1);

    spdlog::info("Found {} IPU devices.", devices.size());
    auto it =
        std::find_if(devices.begin(), devices.end(),
                     [](poplar::Device &device) { return device.attach(); });
    if (it == devices.end()) {
      spdlog::error("Failed to attach to IPU.");
      return false;
    }

    device = std::make_unique<poplar::Device>(std::move(*it));
    spdlog::info("Attached to IPU: arch={}, system={}, numTiles={}",
                 target->getTargetArchString().cloneAsString(),
                 target->getTargetSystemString(), target->getNumTiles());
  }
  target = std::make_unique<poplar::Target>(device->getTarget());

  return true;
}

IPURuntime::IPURuntime() {}
IPURuntime::~IPURuntime() { unload(); }

void IPURuntime::unload() {
  engine.reset();
  target.reset();
  loadedKernel = nullptr;

  if (device) {
    device->detach();
    device.reset();
  }
}

void IPURuntime::load(Kernel &kernel) {
  loadedKernel = dynamic_cast<IPUKernel *>(&kernel);
  if (!loadedKernel) {
    throw std::runtime_error("Kernel is not an IPU kernel.");
  }

  if (!attach(loadedKernel->ipuTarget(), loadedKernel->numTiles())) {
    throw std::runtime_error("Failed to attach to IPU.");
  }

  engine = std::make_unique<::poplar::Engine>(loadedKernel->executable());
  engine->load(*device);
  spdlog::debug("Loaded kernel on IPU");
}

void IPURuntime::execute(size_t num_elements, void *inputs, void *outputs) {
  spdlog::debug("Running kernel on IPU");

  char *inputDataStart = reinterpret_cast<char *>(inputs);
  char *outputDataStart = reinterpret_cast<char *>(outputs);

  char *inputDataEnd = inputDataStart + num_elements *
                                            loadedKernel->numFeatures() *
                                            loadedKernel->bytesPerFeature();
  char *outputDataEnd = outputDataStart + num_elements *
                                              loadedKernel->numResults() *
                                              loadedKernel->bytesPerResult();

  // Each time data is copied to/from the stream the pointer for the next batch,
  // the pointer for the next transfer is incremented within the bounds of the
  // buffer.
  engine->connectStream("inputFIFO", inputDataStart, inputDataEnd);
  engine->connectStream("outputFIFO", outputDataStart, outputDataEnd);

  size_t batch = 0;
  poplar::Engine::TimerTimePoint startTimePoint = engine->getTimeStamp();
  for (size_t i = 0; i < num_elements; i += loadedKernel->batchSize()) {
    engine->run(loadedKernel->programId(), "spnc_kernel");
    batch++;
  }
  poplar::Engine::TimerTimePoint endTimePoint = engine->getTimeStamp();
  spdlog::info(
      "Executed kernel {} times for {} elements (batch size: {}) in {}", batch,
      num_elements, loadedKernel->batchSize(),
      engine->reportTiming(startTimePoint, endTimePoint));
}