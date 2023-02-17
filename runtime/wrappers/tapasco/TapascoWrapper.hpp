#pragma once

#include <vector>
#include <tapasco.hpp>

#include "runtime/src/Executable.h"

#include "pipeline/steps/hdl/ControllerDescription.hpp"


namespace spnc_rt::tapasco_wrapper {

template <class T>
T roundN(const T& n, const T& N) {
  if (n % N == 0)
    return n;
  return n + (N - n % N);
}

class TapascoSPNDevice {
  Kernel kernel;
  ::spnc::ControllerDescription controllerDescription;

  tapasco::Tapasco tap;
  size_t kernelId;
  size_t peCount;

  // we don't want to allocate memory every execution if we don't have to
  std::vector<uint8_t> inputBuffer;
  std::vector<uint8_t> outputBuffer;

  void setInputBuffer(size_t numElements, const void *inputs);
  void resizeOutputBuffer(size_t numElements);
  void execute();
public:
  // these functions can fail
  TapascoSPNDevice(const Kernel& kernel, const ::spnc::ControllerDescription& controllerDescription);
  void execute_query(size_t numElements, const void *inputs, void *outputs);
};

static std::unique_ptr<TapascoSPNDevice> device;

}

namespace spnc_rt {

tapasco_wrapper::TapascoSPNDevice *initTapasco(const Kernel& kernel, const ::spnc::ControllerDescription& controllerDescription) {
  using namespace tapasco_wrapper;

  if (!device)
    device = std::make_unique<TapascoSPNDevice>(kernel, controllerDescription);

  return device.get();
}

}