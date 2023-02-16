cmake_minimum_required(VERSION 3.15)
project(rofl)

set(CMAKE_CXX_STANDARD 17)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_SV_ROS "Build ros" ON)
option(BUILD_SV_LTO "Build lto" OFF)
option(BUILD_SV_PLAY "Build play" OFF)
option(BUILD_SV_TEST "Build test" OFF)
option(BUILD_SV_BENCH "Build bench" OFF)
option(BUILD_SV_NATIVE "Build with -march=native" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CC_NAMESPACE sv)
include(CMakeHelpers)

add_library(sv_options INTERFACE)
include(CompilerWarnings)
enable_warnings(sv_options)
include(Sanitizers)
enable_sanitizers(sv_options)
include(Ccache)

if(BUILD_SV_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(BUILD_SV_NATIVE)
  target_compile_options(sv_options INTERFACE "-march=native")
endif()

find_package(
  catkin QUIET
  COMPONENTS roscpp
             pcl_ros
             tf2_ros
             tf2_eigen
             cv_bridge
             image_transport
             pcl_conversions
             message_filters
             nav_msgs
             sensor_msgs
             visualization_msgs)

find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(absl REQUIRED)
find_package(Boost REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

if(NOT catkin_FOUND)
  set(BUILD_SV_ROS OFF)
  message(WARNING "BUILD_SV_ROS set to OFF because catkin not found.")
endif()

if(BUILD_SV_ROS)
  catkin_package()
endif()

enable_testing()
add_subdirectory(sv)
