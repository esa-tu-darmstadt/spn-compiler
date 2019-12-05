add_library(spnc-rt SHARED
        src/Kernel.cpp)

set_target_properties(spnc-rt PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)

target_include_directories(spnc
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(spnc-rt PRIVATE elf)