include($ENV{TAPASCO_HOME_RUNTIME}/cmake/Tapasco.cmake NO_POLICY_SCOPE)

# TaPaSCo Setup
if(NOT TARGET tapasco)
  #find_package(TapascoTLKM REQUIRED)
  #find_package(TapascoCommon REQUIRED)
  #find_package(TapascoPlatform REQUIRED)
  #find_package(Tapasco REQUIRED)
  find_package(Tapasco REQUIRED)
endif(NOT TARGET tapasco)


add_library(
  tapasco-wrapper
  STATIC
  TapascoWrapper.cpp
)


target_include_directories(
  tapasco-wrapper
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include
)

target_link_libraries(tapasco-wrapper tapasco ${CMAKE_THREAD_LIBS_INIT})