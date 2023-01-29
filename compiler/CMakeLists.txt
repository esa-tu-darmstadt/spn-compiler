# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

find_package(CIRCT REQUIRED CONFIG)
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

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
        src/pipeline/Pipeline.cpp
        src/option/Options.cpp
        src/Driver.cpp
        src/util/Logging.cpp
        src/option/GlobalOptions.cpp
        src/pipeline/steps/codegen/EmitObjectCode.cpp
        src/pipeline/steps/linker/ClangKernelLinking.cpp
        src/toolchain/MLIRToolchain.cpp
        src/toolchain/CPUToolchain.cpp
        src/pipeline/steps/mlir/conversion/HiSPNtoLoSPNConversion.cpp
        src/pipeline/steps/mlir/conversion/LoSPNtoCPUConversion.cpp
        src/pipeline/steps/mlir/conversion/CPUtoLLVMConversion.cpp
        src/pipeline/steps/mlir/conversion/MLIRtoLLVMIRConversion.cpp
        src/pipeline/steps/mlir/transformation/LoSPNTransformations.cpp
        src/pipeline/steps/frontend/SPFlowToMLIRDeserializer.cpp
        src/pipeline/steps/hdl/CreateIPXACT.cpp
        src/pipeline/steps/hdl/EmitVerilogCode.cpp
        src/pipeline/steps/hdl/EmbedController.cpp
        src/pipeline/steps/mlir/conversion/LoSPNtoFPGAConversion.cpp
        src/toolchain/FPGAToolchain.cpp
        ${CAPNP_SRCS}
        )

set(CAPNP_HDR_DIRS "")
if (CAPNP_HDRS)
    # Append directories of all Cap'n'Proto headers.
    foreach(HDR ${CAPNP_HDRS})
        get_filename_component(hdr_dir ${HDR} DIRECTORY)
        list(APPEND CAPNP_HDR_DIRS ${hdr_dir})
    endforeach(HDR ${CAPNP_HDRS})
endif(CAPNP_HDRS)

message(STATUS "Cap'n'Proto header directories: ${CAPNP_HDR_DIRS}")

target_include_directories(spnc
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${CIRCT_INCLUDE_DIRS}
        ${SPN_DIALECT_INCLUDE_DIRS}
        ${CAPNP_HDR_DIRS}
        ${CAPNPC_OUTPUT_DIR})

if (${CUDA_GPU_SUPPORT})
    #
    # If support for CUDA GPUs is enabled, add additinal source files, header directores and libraries.
    #

    # Additional sources for CUDA GPU support
    target_sources(spnc PRIVATE
            src/pipeline/steps/mlir/conversion/CUDA/CUDASerializeToCubin.cpp
            src/pipeline/steps/mlir/conversion/GPUtoLLVMConversion.cpp
            src/pipeline/steps/mlir/conversion/LoSPNtoGPUConversion.cpp
            src/toolchain/CUDAGPUToolchain.cpp
            )

    # Additional include directories.
    target_include_directories(spnc PRIVATE ${CUDA_INCLUDE_DIRS})

    # Additional libraries to link
    target_link_libraries(spnc PRIVATE ${CUDA_RUNTIME_LIBRARY} MLIRLoSPNtoGPUTransforms MLIRNVVMToLLVMIRTranslation)

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
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSupport
        MLIRExecutionEngine

        MLIRLoSPNtoFPGATransforms
        
        CIRCTExportVerilog
        CIRCTHW
        CIRCTHWArith
        CIRCTComb
        CIRCTScheduling
        CIRCTSeq
        CIRCTSeqTransforms
        CIRCTImportFIRFile
        )

llvm_map_components_to_libnames(llvm_libs bitwriter native passes)
target_link_libraries(spnc PRIVATE spnc-common ${llvm_libs} ${mlir_libs} spdlog::spdlog
        CapnProto::capnp cpu_targetinfo stdc++fs)

target_compile_options(spnc PRIVATE ${SPNC_CXX_WARNING_LEVEL})

mlir_check_link_libraries(spnc)

doxygen_doc(TARGET_NAME spnc
        SRC_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDS
        spnc-common mlir-doc
        EXCLUDES
        ../common/include/util/json.hpp)

