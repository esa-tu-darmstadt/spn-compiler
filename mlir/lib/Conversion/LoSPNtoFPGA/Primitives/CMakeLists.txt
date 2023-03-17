add_executable(primitives main.cpp Primitives.cpp)

target_link_libraries(primitives PRIVATE ${LIBS} CIRCTHW CIRCTScheduling CIRCTSeq CIRCTFIRRTL)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

mlir_check_all_link_libraries(lo2hw)