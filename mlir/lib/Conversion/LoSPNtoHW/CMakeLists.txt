add_mlir_conversion_library(MLIRLoSPNtoHWTransforms
    LoSPNtoHWPass.cpp
    #main.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

    LINK_LIBS PUBLIC
    MLIRLoSPN
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRMathDialect
)

target_link_libraries(MLIRLoSPNtoHWTransforms PRIVATE)

#add_executable(lo2hw main.cpp)

#target_include_directories(lo2hw PRIVATE ${LLVM_INCLUDE_DIRS})
#target_link_libraries(lo2hw spnc spnc-rt dl spdlog::spdlog spnc-common)
#add_compile_definitions(TEST_KERNEL_DIR="${CMAKE_CURRENT_BINARY_DIR}")

#set(SPNC_DRIVER_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/driver" CACHE INTERNAL "")