add_library(cpu_targetinfo
        
        TargetInformation.cpp
        )

target_include_directories(cpu_targetinfo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cpu_targetinfo MLIRIR)