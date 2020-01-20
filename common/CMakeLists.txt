add_library(spnc-common INTERFACE)

target_include_directories(spnc-common
        INTERFACE
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)