add_executable(driver src/main.cpp)

target_link_libraries(driver spnc spnc-rt dl)
add_dependencies(driver compiler-rt)
add_compile_definitions(TEST_KERNEL_DIR="${CMAKE_CURRENT_BINARY_DIR}")

add_library(dynamic-load-test SHARED examples/dynamic-load.cpp)