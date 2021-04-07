find_package(pybind11 REQUIRED)

pybind11_add_module(spncpy MODULE src/python-interface.cpp)

target_link_libraries(spncpy PUBLIC spnc-rt spnc-common spnc)

target_include_directories(spncpy
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)


# Create the setup.py and supplement some information from the CMake configure and generate stage.
# This is a multi-stage process, because we first need to use configure_file to supplement 
# configuration variables (used for the MLIR CUDA runtime wrapper library) and then need 
# to substiture the generator expressions used to add target information to the setup.py
if (${CUDA_GPU_SUPPORT} AND NOT ${BUILD_SHARED_LIBS})
    set(INCLUDE_MLIR_CUDA_WRAPPERS "True")
else()
    set(INCLUDE_MLIR_CUDA_WRAPPERS "False")
endif()

get_filename_component(MLIR_CUDA_WRAPPER_NAME ${MLIR_CUDA_WRAPPERS} NAME)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py.in)

file(GENERATE OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/setup.py INPUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py.in)