

include($ENV{TAPASCO_HOME_RUNTIME}/cmake/Tapasco.cmake NO_POLICY_SCOPE)

# TaPaSCo Setup
if(NOT TARGET tapasco)
  #find_package(TapascoTLKM REQUIRED)
  #find_package(TapascoCommon REQUIRED)
  #find_package(TapascoPlatform REQUIRED)
  #find_package(Tapasco REQUIRED)
  find_package(Tapasco REQUIRED)
endif(NOT TARGET tapasco)

# library

add_library(
  tapasco-wrapper
  TapascoWrapper.cpp
)

set_property(TARGET tapasco-wrapper PROPERTY CXX_STANDARD 20)

find_package(spdlog REQUIRED)

target_link_libraries(tapasco-wrapper PRIVATE elf spdlog::spdlog tapasco ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(
  tapasco-wrapper
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include
  ${CMAKE_CURRENT_SOURCE_DIR}/bitpacker/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../compiler/src
)

# executable

add_executable(
  tapasco-app
  TapascoWrapper.cpp
  TapascoApp.cpp
)

set_property(TARGET tapasco-app PROPERTY CXX_STANDARD 20)

find_package(spdlog REQUIRED)

target_link_libraries(tapasco-app PRIVATE elf spdlog::spdlog tapasco ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(
  tapasco-app
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include
  ${CMAKE_CURRENT_SOURCE_DIR}/bitpacker/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../compiler/src
)