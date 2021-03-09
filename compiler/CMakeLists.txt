
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# TODO: The old FindCUDA is deprecated, the new FindCUDAToolkit is only available in version > 3.17.
# Use this feature here after upgrading the required CMake version.
# TODO Guard this with a flag to enable usage of the compiler on devices with no CUDA GPU.
find_package(CUDA)
if (NOT CUDA_FOUND)
    message(FATAL_ERROR "Targeting CUDA GPUs requires a working CUDA install, try setting CUDA_TOOLKIT_ROOT_DIR")
else ()
    message(STATUS "Using CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} from ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "Using CUDA runtime headers: " ${CUDA_INCLUDE_DIRS})
endif ()

# When compiling for CUDA GPUs, the compiler will invoke the PTX compiler and linker
# through the CUDA runtime library API.
find_library(CUDA_RUNTIME_LIBRARY cuda)
if (NOT CUDA_RUNTIME_LIBRARY)
    message(FATAL_ERROR "CUDA runtime library not found. Set location manually through CUDA_RUNTIME_LIBRARY")
else ()
    message(STATUS "Using CUDA runtime library: " ${CUDA_RUNTIME_LIBRARY})
endif ()

# When compiling for CUDA GPUs, the LLVM IR generated for GPU kernels must be linked
# with libdevice, a Nvidia-provided bitcode library with optimized device functions (e.g. log, exp).
if (NOT SPNC_LIBDEVICE)
    # The libdevice was not explicitly specified by the user.
    message(STATUS "Searching " ${CUDA_TOOLKIT_ROOT_DIR} " for libdevice")
    find_file(SPNC_LIBDEVICE libdevice.10.bc HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES "nvvm/libdevice")
    if (NOT SPNC_LIBDEVICE)
        # We did not find it with its modern name, try to find it
        # using a regex matching the names used in older versions of CUDA
        file(GLOB_RECURSE LIBDEVICE_CANDIDATES CONFIGURE_DEPENDS ${CUDA_TOOLKIT_ROOT_DIR}/libdevice*.bc)
        if (LIBDEVICE_CANDIDATES)
            list(GET LIBDEVICE_CANDIDATES 0 SPNC_LIBDEVICE)
        else ()
            message(FATAL_ERROR "Could not find libdevice, specify by defining SPNC_LIBDEVICE")
        endif ()
    endif ()
endif ()
message(STATUS "Using CUDA libdevice from: " ${SPNC_LIBDEVICE})


# When compiling for CUDA GPUs, the generated Kernel must be linked with the
# MLIR CUDA runtime wrappers for data-transfer, kernel-launch etc.
find_library(MLIR_CUDA_RUNTIME_WRAPPERS cuda-runtime-wrappers
        HINTS ${LLVM_BUILD_LIBRARY_DIR})
if (NOT MLIR_CUDA_RUNTIME_WRAPPERS)
    message(FATAL_ERROR "MLIR CUDA runtime wrappers not found")
else ()
    message(STATUS "Using MLIR CUDA wrappers from:" ${MLIR_CUDA_RUNTIME_WRAPPERS})
endif ()

find_package(CapnProto REQUIRED CONFIG)
message(STATUS "Using capnproto version ${CapnProto_VERSION}")
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CAPNPC_SRC_PREFIX ${CMAKE_SOURCE_DIR})

capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS ${CAPNPC_SRC_PREFIX}/xspn/xspn/serialization/binary/capnproto/spflow.capnp)
message(STATUS "Capnproto headers: ${CAPNP_HDRS}")

add_library(spnc SHARED
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
        src/driver/toolchain/CUDAGPUToolchain.cpp
        src/codegen/mlir/pipeline/SPNDialectPipeline.cpp
        src/codegen/mlir/conversion/HiSPNtoLoSPNConversion.cpp
        src/codegen/mlir/conversion/LoSPNtoCPUConversion.cpp
        src/codegen/mlir/conversion/LoSPNtoGPUConversion.cpp
        src/codegen/mlir/conversion/SPNtoStandardConversion.cpp
        src/codegen/mlir/conversion/CPUtoLLVMConversion.cpp
        src/codegen/mlir/conversion/GPUtoLLVMConversion.cpp
        src/codegen/mlir/conversion/MLIRtoLLVMIRConversion.cpp
        src/codegen/mlir/analysis/CollectGraphStatistics.cpp
        src/codegen/mlir/transformation/LoSPNTransformations.cpp
        src/codegen/mlir/frontend/MLIRDeserializer.cpp
        ${CAPNP_SRCS}
        )

target_include_directories(spnc
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${SPN_DIALECT_INCLUDE_DIRS}
        ${CAPNP_HDRS}
        ${CAPNPC_OUTPUT_DIR}
        ${CUDA_INCLUDE_DIRS})

if (MLIR_CUDA_RUNTIME_WRAPPERS)
    get_filename_component(MLIR_CUDA_RUNTIME_WRAPPERS_DIR ${MLIR_CUDA_RUNTIME_WRAPPERS} DIRECTORY)
    message(STATUS "Directory containing the MLIR CUDA runtime wrappers: " ${MLIR_CUDA_RUNTIME_WRAPPERS_DIR})
    target_compile_definitions(spnc PRIVATE SPNC_CUDA_RUNTIME_WRAPPERS_DIR=\"${MLIR_CUDA_RUNTIME_WRAPPERS_DIR}\")
endif ()

if (SPNC_LIBDEVICE)
    target_compile_definitions(spnc PRIVATE SPNC_LIBDEVICE_FILE=\"${SPNC_LIBDEVICE}\")
endif ()

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
        MLIRLoSPNtoGPUTransforms
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
        CapnProto::capnp ${CUDA_RUNTIME_LIBRARY})

mlir_check_link_libraries(spnc)

doxygen_doc(TARGET_NAME spnc
        SRC_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDS
        spnc-common mlir-doc
        EXCLUDES
        ../common/include/util/json.hpp)

