//
// This file is part of the SPNC project.
// Copyright (c) 2021 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "CUDATargetInformation.h"
#include "cuda.h"

inline void emit_cuda_error(const llvm::Twine& message, const char* buffer,
                            CUresult error, mlir::Location loc) {
  mlir::emitError(loc, message.concat(" failed with error code ")
      .concat(llvm::Twine{error})
      .concat("[")
      .concat(buffer)
      .concat("]").str());
}

#define RETURN_ON_CUDA_ERROR(expr, msg)                                        \
  {                                                                            \
    auto _cuda_error = (expr);                                                 \
    if (_cuda_error != CUDA_SUCCESS) {                                         \
      emit_cuda_error(msg, jitErrorBuffer, _cuda_error, loc);                  \
      return {};                                                               \
    }                                                                          \
  }

unsigned int mlir::spn::CUDATargetInformation::maxSharedMemoryPerBlock(mlir::Location loc) {
  // Text buffer to hold error messages if necessary.
  char jitErrorBuffer[4096] = {0};

  // Retrieve information about the maximum amount of shared memory per block for the GPU
  // hosted in this machine from the CUDA device driver API.
  // If multiple devices are present, arbitrarily choose the first
  // one to retrieve information from.

  RETURN_ON_CUDA_ERROR(cuInit(0), "cuInit");
  int numDevices;
  RETURN_ON_CUDA_ERROR(cuDeviceGetCount(&numDevices), "cuDeviceGetCount");

  if (numDevices == 0) {
    mlir::emitWarning(loc, "Found no CUDA devices, assuming zero shared memory");
    return 0;
  }
  if (numDevices > 1) {
    mlir::emitWarning(loc, "Found multiple CUDA devices, retrieving device information from first device");
  }
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  int sharedMem;
  RETURN_ON_CUDA_ERROR(cuDeviceGetAttribute(&sharedMem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                            device), "cuDeviceGetAttribute (Max. shared memory)");
  mlir::emitRemark(loc, Twine("Device supports a maximum of ")
      .concat(Twine{sharedMem})
      .concat(" bytes shared memory per block"));
  return (unsigned) sharedMem;
}