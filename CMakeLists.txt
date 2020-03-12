cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(spnc VERSION 0.0.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(spdlog REQUIRED)

include(Doxygen.cmake)
doxygen_setup()

add_subdirectory(common)
add_subdirectory(compiler)
add_subdirectory(compiler-rt)
add_subdirectory(execute)
add_subdirectory(runtime)
add_subdirectory(python-interface)

doxygen_toplevel()
