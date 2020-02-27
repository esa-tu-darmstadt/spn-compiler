cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(spnc VERSION 0.0.2 LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wfatal-errors -fno-rtti")

add_subdirectory(compiler)
add_subdirectory(execute)
