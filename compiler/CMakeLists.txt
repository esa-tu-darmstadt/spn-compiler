find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src/codegen/mlir/dialects/spn)

add_library(spnc SHARED
        src/driver/option/Options.cpp
        src/Driver.cpp
        src/driver/util/Logging.cpp
        src/driver/option/GlobalOptions.cpp
        src/driver/toolchain/CPUToolchain.cpp
        src/driver/toolchain/MLIRToolchain.cpp
        src/driver/action/LLVMWriteBitcode.cpp
        src/driver/action/LLVMStaticCompiler.cpp
        src/driver/action/LLVMLinker.cpp
        src/driver/action/DetectTracingLib.cpp
        src/driver/action/ClangKernelLinking.cpp
        src/driver/action/MLIRtoLLVMConversion.cpp
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
        src/codegen/mlir/dialects/spn/SPNDialect.cpp
        src/codegen/mlir/dialects/spn/SPNAttributes.cpp
        src/codegen/mlir/dialects/spn/SPNOpInterfaces.cpp
        src/codegen/mlir/MLIRCodeGen.cpp
        src/codegen/mlir/MLIRBodyGen.cpp
        src/codegen/mlir/transform/pattern/CanonicalizationPatterns.cpp
        src/codegen/mlir/transform/pattern/SimplificationPatterns.cpp
        src/codegen/mlir/transform/passes/SPNOpSimplifier.cpp
        src/codegen/mlir/pipeline/MLIRPipeline.cpp
        src/codegen/mlir/lowering/patterns/LowerSPNtoStandardPatterns.cpp
        src/codegen/mlir/lowering/passes/SPNtoStandardLoweringPass.cpp
        src/codegen/mlir/lowering/patterns/LowerSPNtoLLVMPatterns.cpp
        src/codegen/mlir/lowering/passes/SPNToLLVMLoweringPass.cpp)

message(STATUS "MLIR Include:" ${MLIR_INCLUDE_DIRS})

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
        ${CMAKE_CURRENT_BINARY_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(mlir_libs
        ${dialect_libs}
        ${conversion_libs}
        MLIRQuantizerTransforms
        MLIROptLib
        MLIRExecutionEngine
        )

llvm_map_components_to_libnames(llvm_libs bitwriter native passes)
target_link_libraries(spnc PRIVATE spnc-common ${mlir_libs} ${llvm_libs} spdlog::spdlog)

add_dependencies(spnc SPNOpsIncGen SPNOpInterfaceGen)

doxygen_doc(TARGET_NAME spnc
        SRC_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDS
        spnc-common
        EXCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/json/json.hpp)

# CMake setup for a spnc-opt, a tool to run SPN-dialect passes, similar to mlir-opt.
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_llvm_executable(spnc-opt src/tools/opt/spnc-opt.cpp)

llvm_update_compile_flags(spnc-opt)

target_include_directories(spnc-opt
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(spnc-opt PRIVATE ${mlir_libs} ${llvm_libs} spnc)

# Update the path to llvm-lit (taken from llvm/cmake/modules/AddLLVM.cmake)
set(LLVM_EXTERNAL_LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit" CACHE STRING "Command used to spawn lit")
add_subdirectory(test)
