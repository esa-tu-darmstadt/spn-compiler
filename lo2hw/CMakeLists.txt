find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

find_package(CIRCT REQUIRED CONFIG)
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/Dialect)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/Conversion)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/Conversion)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../mlir/include/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../mlir/include/Dialect)

message(${CMAKE_CURRENT_SOURCE_DIR}/../mlir/include/Dialect)
message(${CMAKE_CURRENT_BINARY_DIR}/../mlir/include/Dialect)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

link_directories(${CIRCT_BUILD_LIBRARY_DIR})
add_definitions(${CIRCT_DEFINITIONS})

add_executable(lo2hw lo2hw.cpp mlirgen.cpp rewrite.cpp lo2hwPass.cpp conversion.cpp)
#add_executable(lo2hw test.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRHiSPN
    MLIRLoSPN
    MLIROptLib
    cpu_targetinfo
)

message("Linking " ${LIBS} CIRCTHW)
target_link_libraries(lo2hw PRIVATE ${LIBS} CIRCTHW)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

mlir_check_all_link_libraries(lo2hw)