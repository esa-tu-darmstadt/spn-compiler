add_library(spnc-rt SHARED
        src/runtime.cpp
        src/Executable.cpp)

target_include_directories(spnc-rt
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(spnc-rt PRIVATE elf spnc-common)