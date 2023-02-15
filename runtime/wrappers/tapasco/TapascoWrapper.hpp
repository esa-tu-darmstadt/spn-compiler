#pragma once

#include <vector>
#include <tapasco.hpp>

#include "runtime/src/Executable.h"


namespace spnc_rt::tapasco_wrapper {

class TapascoSPNDevice {
  tapasco::Tapasco tap;
  size_t kernelId;
  size_t peCount;
  Kernel kernel;
  // we don't want to allocate memory every execution if we don't have to
  std::vector<char> inputBuffer;
  std::vector<char> outputBuffer;

  void fillInputBuffer(void* input_ptr, void* aligned_input_ptr,
                       int64_t input_offset, int64_t input_size_dim1,
                       int64_t input_size_dim2, int64_t input_stride_dim1,
                       int64_t input_stride_dim2);


  void execute();
public:
  // these functions can fail
  TapascoSPNDevice(const Kernel& kernel);
  void execute_query(void* input_ptr,
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
                     int64_t output_stride_dim2);
};

static std::unique_ptr<TapascoSPNDevice> device;

static void tapasco_kernel_func(void* input_ptr,
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
                                int64_t output_stride_dim2);

}

namespace spnc_rt {

kernel_function tapasco_get_kernel_func(const Kernel& kernel) {
  using namespace tapasco_wrapper;

  if (!device) {
    try {
      device = std::make_unique<TapascoSPNDevice>(kernel);
    } catch (const std::exception& e) {
      return nullptr;
    }
  }

  return tapasco_wrapper::tapasco_kernel_func;
}

}