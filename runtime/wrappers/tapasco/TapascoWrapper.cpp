#include "TapascoWrapper.hpp"

#include <cassert>
#include "bitpacker/include/bitpacker/bitpacker.hpp"
#include "spdlog/spdlog.h"
#include "packing.hpp"


namespace spnc_rt::tapasco_wrapper {

TapascoSPNDevice::TapascoSPNDevice(const Kernel& kernel): kernel(kernel) {
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

void TapascoSPNDevice::fillInputBuffer(void* input_ptr,
                                       void* aligned_input_ptr,
                                       int64_t input_offset,
                                       int64_t input_size_dim1,
                                       int64_t input_size_dim2,
                                       int64_t input_stride_dim1,
                                       int64_t input_stride_dim2) {
  
  pack<uint32_t>(
    reinterpret_cast<const uint32_t *>(input_ptr),
    reinterpret_cast<const uint32_t *>(input_ptr) + 123,
    8,
    inputBuffer
  );

}

void TapascoSPNDevice::execute() {
  WrappedPointer<char> inputPtr = makeWrappedPointer(inputBuffer.data(), inputBuffer.size());
  InOnly<char> inOnly = makeInOnly(inputPtr);
  size_t inSize = inputBuffer.size();

  WrappedPointer<char> outputPtr = makeWrappedPointer(outputBuffer.data(), output.size());
  OutOnly<char> outOnly = makeOutOnly(outputPtr);
  size_t outSize = output.size();


  /*
      val status             = rf.getRegAtAddress(0x000)
      val retVal             = rf.getRegAtAddress(0x010)
      val loadBaseAddress    = rf.getRegAtAddress(0x020)
      val numLdTransfers     = rf.getRegAtAddress(0x030)
      val storeBaseAddress   = rf.getRegAtAddress(0x040)
      val numSdTransfers     = rf.getRegAtAddress(0x050)
   */

  size_t cycleCounter;
  RetVal<size_t> retVal(cycleCounter);

  tap.launch(
    kernelId,
    retVal,
    inOnly, inSize,
    outOnly, outSize
  );
}

void TapascoSPNDevice::execute_query(void* input_ptr,
                                     void* aligned_input_ptr,
                                     int64_t input_offset,
                                     int64_t input_size_dim1,
                                     int64_t input_size_dim2,
                                     int64_t input_stride_dim1,
                                     int64_t input_stride_dim2,
                                     void* output_ptr,
                                     void* output_aligned_ptr,
                                     int64_t output_offset,
                                     int64_t output_size_dim1,
                                     int64_t output_size_dim2,
                                     int64_t output_stride_dim1,
                                     int64_t output_stride_dim2) {
  using namespace tapasco;
  assert(false && "not implemented");


  


  execute();  
}

void tapasco_kernel_func(void* input_ptr,
                         void* aligned_input_ptr,
                         int64_t input_offset,
                         int64_t input_size_dim1,
                         int64_t input_size_dim2,
                         int64_t input_stride_dim1,
                         int64_t input_stride_dim2,
                         void* output_ptr,
                         void* output_aligned_ptr,
                         int64_t output_offset,
                         int64_t output_size_dim1,
                         int64_t output_size_dim2,
                         int64_t output_stride_dim1,
                         int64_t output_stride_dim2) {
  assert(device && "device is not initialized");

  device->execute_query(
    input_ptr,
    aligned_input_ptr,
    input_offset,
    input_size_dim1,
    input_size_dim2,
    input_stride_dim1,
    input_stride_dim2,
    output_ptr,
    output_aligned_ptr,
    output_offset,
    output_size_dim1,
    output_size_dim2,
    output_stride_dim1,
    output_stride_dim2
  );
}

}