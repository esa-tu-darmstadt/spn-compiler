#include "TapascoWrapper.hpp"

#include <cassert>
#include "bitpacker/include/bitpacker/bitpacker.hpp"
#include <spdlog/spdlog.h>
#include "packing.hpp"


namespace spnc_rt::tapasco_wrapper {

TapascoSPNDevice::TapascoSPNDevice(const Kernel& kernel):
  kernel(kernel) {

  using namespace tapasco;

  // TODO: Add functionality to query device information to verify that
  // the correct kernel is loaded.

  if (kernel.getKernelType() != KernelType::FPGA_KERNEL)
    throw std::runtime_error("wrong kernel type");

  fpgaKernel = kernel.getFPGAKernel();
  kernelId = fpgaKernel.kernelId;

  spdlog::info("Trying to initialize kernel with id {}", kernelId);

  peCount = tap.kernel_pe_count(kernelId);

  if (peCount == 0)
    throw std::runtime_error("not PE with the provided kernel id found");

  spdlog::info("Using {} PEs with id {}", peCount, kernelId);
  spdlog::info("The accelerator runs at frequency {}Mhz", tap.design_frequency());
}

void TapascoSPNDevice::setInputBuffer(size_t numElements, const void *inputs) {
  size_t inByteSize = roundN<size_t>(fpgaKernel.spnVarCount * fpgaKernel.spnBitsPerVar, 8) / 8;
  inputBuffer.resize(numElements * inByteSize);

  const uint32_t *in = reinterpret_cast<const uint32_t *>(inputs);

  pack(
    in,
    in + numElements * fpgaKernel.spnVarCount,
    fpgaKernel.spnBitsPerVar,
    inputBuffer
  );

  size_t roundedSize = roundN<size_t>(inputBuffer.size(), fpgaKernel.memDataWidth / 8);
  inputBuffer.resize(roundedSize);
}

void TapascoSPNDevice::resizeOutputBuffer(size_t numElements) {
  size_t outByteSize = roundN<size_t>(fpgaKernel.spnResultWidth, 8) / 8;
  size_t s = roundN<size_t>(numElements * outByteSize, fpgaKernel.memDataWidth / 8);
  outputBuffer.resize(s);
}

void TapascoSPNDevice::executeQuery(size_t numElements, const void *inputs, void *outputs) {
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

  size_t loadBeatCount = inSize / (fpgaKernel.memDataWidth / 8);
  size_t storeBeatCount = outSize / (fpgaKernel.memDataWidth / 8);

  spdlog::info(
    "Would execute Tapasco with {} load beats and {} save beats (numElements = {})",
    loadBeatCount,
    storeBeatCount,
    numElements
  );

  return;

  auto job = tap.launch(
    kernelId, retVal, inOnly, loadBeatCount,
    outOnly, storeBeatCount
  );

  if (job() != TAPASCO_SUCCESS)
    spdlog::error("Tapasco job failed");

  spdlog::info("Tapasco job executed successfully");

  std::copy_n(
    reinterpret_cast<const double *>(outputBuffer.data()),
    numElements,
    reinterpret_cast<double *>(outputs)
  );
}

}

int main() {
  return 0;
}