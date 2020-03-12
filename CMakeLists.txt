cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(spnc VERSION 0.0.2 LANGUAGES CXX)

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
