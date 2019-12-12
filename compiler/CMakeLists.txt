find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


find_package(Boost REQUIRED)

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

add_library(spnc SHARED src/Driver.cpp
        src/graph-ir/GraphIRNode.cpp
        src/graph-ir/InputVar.cpp
        src/graph-ir/Histogram.cpp
        src/graph-ir/WeightedSum.cpp
        src/graph-ir/Sum.cpp
        src/graph-ir/Product.cpp
        src/json/Parser.cpp
        src/transform/BaseVisitor.cpp
        src/util/DotVisitor.cpp
        src/transform/BinaryTreeTransform.cpp
        src/codegen/shared/VectorizationTraversal.cpp
        src/codegen/shared/BFSOrderProducer.cpp
        src/codegen/shared/PackingSolver.cpp
        src/codegen/shared/SchedulingConflictTraversal.cpp
        src/codegen/llvm-ir/LLVMCodegen.cpp
        src/codegen/llvm-ir/IREmitter.cpp)

target_include_directories(spnc
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS}
	${GUROBI_HEADER_DIRS}
	${Boost_INCLUDE_DIR})

llvm_map_components_to_libnames(llvm_libs all)
target_link_libraries(spnc PRIVATE ${llvm_libs} ${GUROBI_LIBS})
