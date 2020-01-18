find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


add_library(spnc SHARED src/Driver.cpp
        src/driver/Kernel.cpp
        src/driver/toolchain/CPUToolchain.cpp
        src/driver/action/LLVMWriteBitcode.cpp
        src/driver/action/LLVMStaticCompiler.cpp
        src/graph-ir/GraphIRNode.cpp
        src/graph-ir/InputVar.cpp
        src/graph-ir/Histogram.cpp
        src/graph-ir/WeightedSum.cpp
        src/graph-ir/Sum.cpp
        src/graph-ir/Product.cpp
        src/json/Parser.cpp
        src/transform/BaseVisitor.cpp
        src/transform/IRTransformationPass.cpp
        src/util/DotVisitor.cpp
        src/transform/BinaryTreeTransform.cpp
        src/codegen/llvm-ir/CPU/LLVMCPUCodegen.cpp
        src/codegen/llvm-ir/CPU/body/CodeGenScalarBody.cpp
        src/codegen/llvm-ir/CPU/loop/CodeGenSerialLoop.cpp)

target_include_directories(spnc
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS})

llvm_map_components_to_libnames(llvm_libs all)
target_link_libraries(spnc PRIVATE ${llvm_libs})

set_target_properties(spnc PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
