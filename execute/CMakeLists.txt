add_executable(driver src/main.cpp)

target_link_libraries(driver spnc spnc-rt)

add_library(dynamic-load-test SHARED examples/dynamic-load.cpp)