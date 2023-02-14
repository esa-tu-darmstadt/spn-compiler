#include "TapascoWrapper.hpp"

#include <cassert>


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
}

void TapascoSPNDevice::fillInputBuffer(void* input_ptr,
                                       void* aligned_input_ptr,
                                       int64_t input_offset,
                                       int64_t input_size_dim1,
                                       int64_t input_size_dim2,
                                       int64_t input_stride_dim1,
                                       int64_t input_stride_dim2) {
  
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

  fillInputBuffer(
    input_ptr, aligned_input_ptr, input_offset,
    input_size_dim1, input_size_dim2,
    input_stride_dim1, input_stride_dim2
  );

  WrappedPointer<char> inputPtr = makeWrappedPointer(inputBuffer.data(), inputBuffer.size());


  
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