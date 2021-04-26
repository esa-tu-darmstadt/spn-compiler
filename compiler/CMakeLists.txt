
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

if (${CUDA_GPU_SUPPORT})
    include(DetectCUDALibs.cmake)
    detect_cuda_libs()
endif (${CUDA_GPU_SUPPORT})

find_package(CapnProto REQUIRED CONFIG)
message(STATUS "Using capnproto version ${CapnProto_VERSION}")
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CAPNPC_SRC_PREFIX ${CMAKE_SOURCE_DIR})

capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS ${CAPNPC_SRC_PREFIX}/xspn/xspn/serialization/binary/capnproto/spflow.capnp)
message(STATUS "Capnproto headers: ${CAPNP_HDRS}")

add_library(spnc
        src/driver/option/Options.cpp
        src/Driver.cpp
        src/driver/util/Logging.cpp
        src/driver/option/GlobalOptions.cpp
        src/driver/action/LLVMWriteBitcode.cpp
        src/driver/action/LLVMStaticCompiler.cpp
        src/driver/action/EmitObjectCode.cpp
        src/driver/action/LLVMLinker.cpp
        src/driver/action/DetectTracingLib.cpp
        src/driver/action/ClangKernelLinking.cpp
        src/driver/toolchain/MLIRToolchain.cpp
        src/driver/toolchain/CPUToolchain.cpp
        src/codegen/mlir/pipeline/SPNDialectPipeline.cpp
        src/codegen/mlir/conversion/HiSPNtoLoSPNConversion.cpp
        src/codegen/mlir/conversion/LoSPNtoCPUConversion.cpp
        src/codegen/mlir/conversion/SPNtoStandardConversion.cpp
        src/codegen/mlir/conversion/CPUtoLLVMConversion.cpp
        src/codegen/mlir/conversion/MLIRtoLLVMIRConversion.cpp
        src/codegen/mlir/transformation/LoSPNTransformations.cpp
        src/codegen/mlir/frontend/MLIRDeserializer.cpp
        ${CAPNP_SRCS}
        )

target_include_directories(spnc
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${SPN_DIALECT_INCLUDE_DIRS}
        ${CAPNP_HDRS}
        ${CAPNPC_OUTPUT_DIR})

if (${CUDA_GPU_SUPPORT})
    #
    # If support for CUDA GPUs is enabled, add additinal source files, header directores and libraries.
    #

    # Additional sources for CUDA GPU support
    target_sources(spnc PRIVATE
            src/codegen/mlir/conversion/GPUtoLLVMConversion.cpp
            src/codegen/mlir/conversion/LoSPNtoGPUConversion.cpp
            src/driver/toolchain/CUDAGPUToolchain.cpp
            )

    # Additional include directories.
    target_include_directories(spnc PRIVATE ${CUDA_INCLUDE_DIRS})

    # Additional libraries to link
    target_link_libraries(spnc PRIVATE ${CUDA_RUNTIME_LIBRARY} MLIRLoSPNtoGPUTransforms)

    if (MLIR_CUDA_WRAPPERS)
        get_filename_component(MLIR_CUDA_RUNTIME_WRAPPERS_DIR ${MLIR_CUDA_WRAPPERS} DIRECTORY)
        message(STATUS "Directory containing the MLIR CUDA runtime wrappers: " ${MLIR_CUDA_RUNTIME_WRAPPERS_DIR})
        target_compile_definitions(spnc PRIVATE SPNC_CUDA_RUNTIME_WRAPPERS_DIR=\"${MLIR_CUDA_RUNTIME_WRAPPERS_DIR}\")
    endif ()

    if (SPNC_LIBDEVICE)
        target_compile_definitions(spnc PRIVATE SPNC_LIBDEVICE_FILE=\"${SPNC_LIBDEVICE}\")
    endif ()
endif (${CUDA_GPU_SUPPORT})

# Set a compiler definition for the CUDA GPU support for use
# of conditional compilation inside source files
if (${CUDA_GPU_SUPPORT})
    target_compile_definitions(spnc PRIVATE SPNC_CUDA_SUPPORT=1)
else (${CUDA_GPU_SUPPORT})
    target_compile_definitions(spnc PRIVATE SPNC_CUDA_SUPPORT=0)
endif (${CUDA_GPU_SUPPORT})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(mlir_libs
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRHiSPN
        MLIRLoSPN
        MLIRHiSPNtoLoSPNTransforms
        MLIRLoSPNtoCPUTransforms
        MLIRSPNtoStandardTransforms
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslation
        MLIRSupport
        MLIRExecutionEngine
        MLIRTargetNVVMIR
        )

llvm_map_components_to_libnames(llvm_libs bitwriter native passes)
target_link_libraries(spnc PRIVATE spnc-common ${llvm_libs} ${mlir_libs} spdlog::spdlog
        CapnProto::capnp cpu_targetinfo)

mlir_check_link_libraries(spnc)

doxygen_doc(TARGET_NAME spnc
        SRC_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDS
        spnc-common mlir-doc
        EXCLUDES
        ../common/include/util/json.hpp)

