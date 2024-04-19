if (${CUDA_GPU_SUPPORT})
    # Only build the runtime wrapper library if CUDA support was requested.
    add_subdirectory(cuda)
endif (${CUDA_GPU_SUPPORT})

if(${TAPASCO_FPGA_SUPPORT})
  add_subdirectory(tapasco)
endif(${TAPASCO_FPGA_SUPPORT})