cc_library(
  NAME base
  DEPS sv_options
  INCDIRS ${CMAKE_SOURCE_DIR})

if(BUILD_SV_TEST)
  cc_library(NAME gtest DEPS GTest::GTest)
  cc_library(NAME gtest_main DEPS GTest::GTest GTest::Main)
endif()

if(BUILD_SV_BENCH)
  cc_library(NAME bench DEPS benchmark::benchmark)
  cc_library(NAME bench_main DEPS benchmark::benchmark
                                  benchmark::benchmark_main)
endif()

add_subdirectory(util)
add_subdirectory(rofl)

if(BUILD_SV_ROS)
  add_subdirectory(ros1)
endif()
