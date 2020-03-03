find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_definitions(${LLVM_DEFINITIONS})

add_library(spnc SHARED src/Driver.cpp
        src/driver/toolchain/CPUToolchain.cpp
        src/driver/action/LLVMWriteBitcode.cpp
        src/driver/action/LLVMStaticCompiler.cpp
        src/driver/action/LLVMLinker.cpp
        src/driver/action/TracingLib.cpp
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
        src/codegen/llvm-ir/transform/NumericalValueTracingPass.cpp)

target_include_directories(spnc
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS})

llvm_map_components_to_libnames(llvm_libs bitwriter passes)
target_link_libraries(spnc PRIVATE ${llvm_libs})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)

add_llvm_library( NumericalValueTracingPass MODULE
        src/codegen/llvm-ir/transform/NumericalValueTracingPass.cpp

        PLUGIN_TOOL
        opt
        )

target_include_directories(NumericalValueTracingPass
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS})