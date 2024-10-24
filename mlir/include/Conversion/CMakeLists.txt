add_subdirectory(HiSPNtoLoSPN)
add_subdirectory(LoSPNtoGPU)
add_subdirectory(LoSPNtoCPU)

if(${IPU_SUPPORT})
    add_subdirectory(CPUtoPoplar)
endif(${IPU_SUPPORT})