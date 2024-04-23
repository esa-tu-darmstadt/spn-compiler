add_mlir_conversion_library(MLIRLoSPNtoFPGATransforms2
    LoSPNtoFPGAPass2.cpp
    Conversion.cpp
    Scheduling.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/mlir/include/Conversion
    ${PROJECT_SOURCE_DIR}/lo2hw

    LINK_LIBS PUBLIC
    MLIRLoSPN
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRMathDialect

    CIRCTHW
    CIRCTHWArith
    CIRCTComb
    CIRCTScheduling
    CIRCTSeq
    CIRCTFIRRTL
)

target_link_libraries(MLIRLoSPNtoFPGATransforms2 PRIVATE firp)

#message("adding include  dir " ${PROJECT_SOURCE_DIR}/lo2hw)

target_include_directories(MLIRLoSPNtoFPGATransforms2 PRIVATE
  ${PROJECT_SOURCE_DIR}/common/include
  ${PROJECT_SOURCE_DIR}/mlir/FIRRTLPP/include
)
#target_include_directories(MLIRLoSPNtoFPGATransforms PRIVATE ${PROJECT_SOURCE_DIR}/lo2hw)

#add_executable(lo2hw main.cpp)

#target_include_directories(lo2hw PRIVATE ${LLVM_INCLUDE_DIRS})
#target_link_libraries(lo2hw spnc spnc-rt dl spdlog::spdlog spnc-common)
#add_compile_definitions(TEST_KERNEL_DIR="${CMAKE_CURRENT_BINARY_DIR}")

#set(SPNC_DRIVER_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/driver" CACHE INTERNAL "")
