//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

// This file is a modified copy of CudaRuntimeWrappers.cpp in mlir/lib/ExecutionEngine in the LLVM repository.

#include <cassert>
#include <numeric>
#include <iostream>

#include "cuda.h"

#ifdef _WIN32
#define MLIR_CUDA_WRAPPERS_EXPORT __declspec(dllexport)
#else
#define MLIR_CUDA_WRAPPERS_EXPORT
#endif // _WIN32

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

// Make the primary context of device 0 current for the duration of the instance
// and restore the previous context on destruction.
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to CUDA primary context for device ordinal 0.
    static CUcontext context = [] {
      CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0));
      CUdevice device;
      CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/0));
      CUcontext ctx;
      // Note: this does not affect the current context.
      CUDA_REPORT_IF_ERROR(cuDevicePrimaryCtxRetain(&ctx, device));
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoad(void* data) {
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  return module;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuModuleUnload(CUmodule module) {
  CUDA_REPORT_IF_ERROR(cuModuleUnload(module));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUfunction
mgpuModuleGetFunction(CUmodule module, const char* name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void** params,
                 void** extra) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUstream mgpuStreamCreate() {
  ScopedContext scopedContext;
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamDestroy(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuStreamSynchronize(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamWaitEvent(CUstream stream,
                                                              CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUevent mgpuEventCreate() {
  ScopedContext scopedContext;
  CUevent event = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  return event;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventDestroy(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventDestroy(event));
}

extern MLIR_CUDA_WRAPPERS_EXPORT "C" void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern MLIR_CUDA_WRAPPERS_EXPORT "C" void mgpuEventRecord(CUevent event,
                                                          CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

extern "C" void* mgpuMemAlloc(uint64_t sizeBytes, CUstream /*stream*/) {
  ScopedContext scopedContext;
  CUdeviceptr ptr;
#if SPNC_CUDA_UNIFIED_MEMORY
  CUDA_REPORT_IF_ERROR(cuMemAllocManaged(&ptr, sizeBytes, CU_MEM_ATTACH_GLOBAL));
#else
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&ptr, sizeBytes));
#endif
  return reinterpret_cast<void*>(ptr);
}

extern "C" void mgpuMemFree(void* ptr, CUstream /*stream*/) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

extern "C" void mgpuMemcpy(void* dst, void* src, uint64_t sizeBytes,
                           CUstream stream) {
#if SPNC_CUDA_UNIFIED_MEMORY
  // Assumes physically shared memory between host CPU and GPU.
#else
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src),
                                     sizeBytes, stream));
#endif
}
