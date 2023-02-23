#pragma once

#include <vector>
#include <tapasco.hpp>

#include "Kernel.h"


namespace spnc_rt::tapasco_wrapper {

using namespace spnc;

template <class T>
T roundN(const T& n, const T& N) {
  if (n % N == 0)
    return n;
  return n + (N - n % N);
}

class TapascoSPNDevice {
  Kernel kernel;
  FPGAKernel fpgaKernel;

  tapasco::Tapasco tap;
  size_t kernelId;
  size_t peCount;

  // we don't want to allocate memory every execution if we don't have to
  std::vector<uint8_t> inputBuffer;
  std::vector<uint8_t> outputBuffer;

  void setInputBuffer(size_t numElements, const void *inputs);
  void resizeOutputBuffer(size_t numElements);
public:
  // these functions can fail
  TapascoSPNDevice(const Kernel& kernel);
  void executeQuery(size_t numElements, const void *inputs, void *outputs);
};

static std::unique_ptr<TapascoSPNDevice> device;

}

namespace spnc_rt {

using namespace spnc;

tapasco_wrapper::TapascoSPNDevice *initTapasco(const Kernel& kernel) {
  using namespace tapasco_wrapper;

  if (!device)
    device = std::make_unique<TapascoSPNDevice>(kernel);

  return device.get();
}

}