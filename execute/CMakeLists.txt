add_executable(driver src/main.cpp)

target_include_directories(driver PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(driver spnc spnc-rt dl spdlog::spdlog)
add_dependencies(driver compiler-rt)
add_compile_definitions(TEST_KERNEL_DIR="${CMAKE_CURRENT_BINARY_DIR}")

set(SPNC_DRIVER_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/driver" CACHE INTERNAL "")