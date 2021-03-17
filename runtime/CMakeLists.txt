add_library(spnc-rt SHARED
        src/runtime.cpp
        src/Executable.cpp)

target_include_directories(spnc-rt
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(spnc-rt PRIVATE elf spnc-common spdlog::spdlog)

option(CPU_PARALLEL_RUNTIME
        "Enable/disable thread-parallel execution of multiple batches on CPU in the runtime"
        ON)

if (${CPU_PARALLEL_RUNTIME})
    find_package(OpenMP REQUIRED)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(spnc-rt PRIVATE OpenMP::OpenMP_CXX)
    endif (OpenMP_CXX_FOUND)
endif (${CPU_PARALLEL_RUNTIME})

doxygen_doc(TARGET_NAME spnc-rt
        SRC_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDS
        spnc-common)