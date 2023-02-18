#include "TapascoWrapper.hpp"

#include <cassert>
#include "bitpacker/include/bitpacker/bitpacker.hpp"
#include <spdlog/spdlog.h>
#include "packing.hpp"


namespace spnc_rt::tapasco_wrapper {

TapascoSPNDevice::TapascoSPNDevice(const Kernel& kernel, const ::spnc::ControllerDescription& controllerDescription):
  kernel(kernel), controllerDescription(controllerDescription) {

  using namespace tapasco;

  // TODO: Add functionality to query device information to verify that
  // the correct kernel is loaded.

  char *end_ptr;
  kernelId = std::strtol(kernel.kernelName().c_str(), &end_ptr, 10);

  if (end_ptr == kernel.kernelName().c_str() || *end_ptr != '\0')
    throw std::runtime_error("kernel function name must be a integer number (kernel id)");

  peCount = tap.kernel_pe_count(kernelId);

  if (peCount == 0)
    throw std::runtime_error("not PE with the provided kernel id found");

  spdlog::info("Using {} PEs with id {}", peCount, kernelId);
  spdlog::info("The accelerator runs at frequency {}Mhz", tap.design_frequency());
}

void TapascoSPNDevice::setInputBuffer(size_t numElements, const void *inputs) {
  // NOTE: We will assume uint32_t for now!

  size_t inByteSize = roundN<size_t>(controllerDescription.spnVarCount * controllerDescription.spnBitsPerVar, 8) / 8;
  inputBuffer.resize(numElements * inByteSize);

  const uint32_t *in = reinterpret_cast<const uint32_t *>(inputs);

  pack(
    in,
    in + numElements * controllerDescription.spnVarCount,
    controllerDescription.spnBitsPerVar,
    inputBuffer
  );

  size_t roundedSize = roundN<size_t>(inputBuffer.size(), controllerDescription.memDataWidth / 8);
  inputBuffer.resize(roundedSize);
}

void TapascoSPNDevice::resizeOutputBuffer(size_t numElements) {
  size_t outByteSize = roundN<size_t>(controllerDescription.spnResultWidth, 8) / 8;
  size_t s = roundN<size_t>(numElements * outByteSize, controllerDescription.memDataWidth / 8);
  outputBuffer.resize(s);
}

void TapascoSPNDevice::execute_query(size_t numElements, const void *inputs, void *outputs) {
  using namespace tapasco;

  setInputBuffer(numElements, inputs);
  resizeOutputBuffer(numElements);

  WrappedPointer<uint8_t> inputPtr = makeWrappedPointer(inputBuffer.data(), inputBuffer.size());
  auto inOnly = makeInOnly(std::move(inputPtr));
  size_t inSize = inputBuffer.size();

  WrappedPointer<uint8_t> outputPtr = makeWrappedPointer(outputBuffer.data(), outputBuffer.size());
  auto outOnly = makeOutOnly(std::move(outputPtr));
  size_t outSize = outputBuffer.size();

  size_t cycleCount = 0;
  RetVal<size_t> retVal{&cycleCount};

  size_t loadBeatCount = inSize / (controllerDescription.memDataWidth / 8);
  size_t storeBeatCount = outSize / (controllerDescription.memDataWidth / 8);

  auto job = tap.launch(
    kernelId, retVal, inOnly, loadBeatCount,
    outOnly, storeBeatCount
  );

  if (job() != TAPASCO_SUCCESS)
    spdlog::error("Tapasco job failed");

  spdlog::info("Tapasco job executed successfully");

  // NOTE: We will assume float32 for now. This should be changed to float64 in the future.
  std::copy_n(
    reinterpret_cast<const float *>(outputBuffer.data()),
    numElements,
    reinterpret_cast<float *>(outputs)
  );
}

}

int main() {
  using namespace ::spnc_rt;
  using namespace ::spnc_rt::tapasco_wrapper;

  Kernel kernel{
    "",
    "123",
    KernelQueryType::JOINT_QUERY,
    KernelTarget::FPGA,
    1,
    5,
    1,
    1,
    4,
    "uint32_t"
  };

  ::spnc::ControllerDescription controllerDescription{
    .spnVarCount = 5,
    .spnBitsPerVar = 8,
    .spnResultWidth = 31,
    .mAxisControllerWidth = 32,
    .sAxisControllerWidth = 40,
    .memDataWidth = 32,
    .memAddrWidth = 32,
    .liteDataWidth = 32,
    .liteAddrWidth = 32
  };

  TapascoSPNDevice *device = initTapasco(kernel, controllerDescription);

  size_t count = 100;

  std::vector<uint32_t> inputs(count, 0);
  std::vector<float> outputs(count);

  device->execute_query(count, inputs.data(), outputs.data());

  spdlog::info("Got");
  for (float d : outputs)
    spdlog::info("{}", d);

  return 0;
}