find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

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
        src/driver/toolchain/CPUToolchain.cpp
        src/driver/action/LLVMWriteBitcode.cpp
        src/driver/action/LLVMStaticCompiler.cpp
        src/driver/action/LLVMLinker.cpp
        src/driver/action/DetectTracingLib.cpp
        src/driver/action/ClangKernelLinking.cpp
        src/graph-ir/GraphIRNode.cpp
        src/graph-ir/InputVar.cpp
        src/graph-ir/Histogram.cpp
        src/graph-ir/WeightedSum.cpp
        src/graph-ir/Sum.cpp
        src/graph-ir/Product.cpp
        src/frontend/json/Parser.cpp
        src/graph-ir/transform/BaseVisitor.cpp
        src/graph-ir/transform/IRTransformationPass.cpp
        src/graph-ir/util/DotVisitor.cpp
        src/graph-ir/util/GraphStatVisitor.cpp
        src/graph-ir/transform/BinaryTreeTransform.cpp
        src/codegen/llvm-ir/CPU/LLVMCPUCodegen.cpp
        src/codegen/llvm-ir/CPU/body/CodeGenScalarBody.cpp
        src/codegen/llvm-ir/CPU/loop/CodeGenSerialLoop.cpp
        src/codegen/llvm-ir/pipeline/LLVMPipeline.cpp
        src/codegen/llvm-ir/transform/NumericalValueTracingPass.cpp
        src/driver/toolchain/MLIRToolchain.cpp
        src/codegen/mlir/codegen/MLIRCodeGen.cpp
        src/codegen/mlir/codegen/MLIRGraphGen.cpp
        src/codegen/mlir/pipeline/SPNDialectPipeline.cpp
        src/codegen/mlir/conversion/SPNtoStandardConversion.cpp
        src/codegen/mlir/conversion/SPNtoLLVMConversion.cpp
        src/codegen/mlir/conversion/MLIRtoLLVMIRConversion.cpp
        src/codegen/mlir/analysis/CollectGraphStatistics.cpp
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
        ${CAPNPC_OUTPUT_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(mlir_libs
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSPN
        MLIRSPNtoStandardTransforms
        MLIRSPNtoLLVMTransforms
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslation
        MLIRSupport
        MLIRExecutionEngine
        )

llvm_map_components_to_libnames(llvm_libs bitwriter native passes)
target_link_libraries(spnc PRIVATE spnc-common ${llvm_libs} ${mlir_libs} spdlog::spdlog CapnProto::capnp)

mlir_check_link_libraries(spnc)

doxygen_doc(TARGET_NAME spnc
        SRC_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDS
        spnc-common mlir-doc
        EXCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/json/json.hpp)

