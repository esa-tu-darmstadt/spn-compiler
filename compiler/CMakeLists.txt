find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_definitions(${LLVM_DEFINITIONS})


set(GUROBI_DIR "$ENV{GUROBI_HOME}")

if(NOT GUROBI_DIR)
  message(FATAL_ERROR "Set env var GUROBI_HOME")
endif()

find_path(GUROBI_HEADER_DIR 
          NAMES gurobi_c++.h
          PATHS "${GUROBI_DIR}/include"
          )

find_library(GUROBI_LIB 
              NAMES gurobi90
              PATHS "${GUROBI_DIR}/lib"
              )

find_library(GUROBI_CXX_LIB 
              NAMES gurobi_c++
              PATHS "${GUROBI_DIR}/lib"
              )
	 

if((NOT GUROBI_HEADER_DIR) OR ((NOT GUROBI_LIB) OR (NOT GUROBI_CXX_LIB)))
  message(FATAL_ERROR "Could not find library or header in GUROBI_HOME")
endif()

set(GUROBI_HEADER_DIRS "${GUROBI_HEADER_DIR}" )
set(GUROBI_LIBS "${GUROBI_LIB};${GUROBI_CXX_LIB}" )


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
        src/graph-ir/Gauss.cpp
        src/graph-ir/WeightedSum.cpp
        src/graph-ir/Sum.cpp
        src/graph-ir/Product.cpp
        src/frontend/json/Parser.cpp
        src/graph-ir/transform/BaseVisitor.cpp
        src/graph-ir/transform/IRTransformationPass.cpp
        src/graph-ir/util/DotVisitor.cpp
        src/graph-ir/util/GraphStatVisitor.cpp
        src/graph-ir/util/GraphIRTools.cpp
        src/graph-ir/util/BFSOrderProducer.cpp
        src/graph-ir/transform/BinaryTreeTransform.cpp
        src/graph-ir/transform/AlternatingNodesTransform.cpp
        src/codegen/llvm-ir/CPU/LLVMCPUCodegen.cpp
        src/codegen/llvm-ir/CPU/body/CodeGenScalarBody.cpp
        src/codegen/llvm-ir/CPU/body/CodeGenVecBody.cpp
        src/codegen/llvm-ir/CPU/body/vec/SchedulingConflictTraversal.cpp
        src/codegen/llvm-ir/CPU/body/vec/CostInfo.cpp
        src/codegen/llvm-ir/CPU/body/vec/PackingSolver.cpp
        src/codegen/llvm-ir/CPU/body/vec/PackingHeuristic.cpp
        src/codegen/llvm-ir/CPU/body/vec/IREmitter.cpp
        src/codegen/llvm-ir/CPU/body/vec/InitialChainBuilder.cpp
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
	${GUROBI_HEADER_DIRS}
        ${LLVM_INCLUDE_DIRS})

llvm_map_components_to_libnames(llvm_libs bitwriter passes x86codegen)
target_link_libraries(spnc PRIVATE ${llvm_libs} spdlog::spdlog ${GUROBI_LIBS})

doxygen_doc(TARGET_NAME spnc
        SRC_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDS
        spnc-common
        EXCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/json/json.hpp)
