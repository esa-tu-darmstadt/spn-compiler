find_package(pybind11 REQUIRED)

pybind11_add_module(spncpy MODULE
        src/python-interface.cpp)

target_link_libraries(spncpy PUBLIC spnc-rt spnc-common spnc)

target_include_directories(spncpy

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)