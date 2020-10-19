cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(spnc VERSION 0.0.4 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(spdlog REQUIRED)

include(Doxygen.cmake)
doxygen_setup()

add_subdirectory(common)
# Ensure processing of ./execute before ./compiler
# Reason: tests in ./compiler depend on CMake variables set in ./execute
add_subdirectory(execute)
add_subdirectory(compiler)
add_subdirectory(compiler-rt)
add_subdirectory(runtime)
add_subdirectory(python-interface)

doxygen_toplevel()
