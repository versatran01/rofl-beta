cc_library(
  NAME util_log
  HDRS "logging.h"
  DEPS sv_base fmt::fmt glog::glog
  DEFINES FMT_DEPRECATED_OSTREAM)

cc_library(
  NAME util_grid2d
  HDRS "grid2d.h"
  DEPS sv_base opencv_core)
cc_test(
  NAME util_grid2d_test
  SRCS "grid2d_test.cpp"
  DEPS sv_util_grid2d sv_gtest_main)

cc_library(
  NAME util_eigen
  SRCS "eigen.cpp"
  DEPS sv_util_log Eigen3::Eigen)
cc_test(
  NAME util_eigen_test
  SRCS "eigen_test.cpp"
  DEPS sv_util_eigen sv_gtest_main sv_bench)
cc_bench(
  NAME util_eigen_bench
  SRCS "eigen_test.cpp"
  DEPS sv_util_eigen sv_bench_main sv_gtest)

cc_library(
  NAME util_tbb
  HDRS "tbb.h"
  DEPS sv_base TBB::tbb)
cc_test(
  NAME util_tbb_test
  SRCS "tbb_test.cpp"
  DEPS sv_util_tbb sv_gtest_main)

cc_library(
  NAME util_timer
  HDRS "timer.h"
  DEPS sv_base absl::time)

cc_library(
  NAME util_summary
  SRCS "summary.cpp"
  DEPS sv_util_log sv_util_timer absl::flat_hash_map)

cc_library(
  NAME util_math
  SRCS "math.cpp"
  DEPS sv_util_eigen)
cc_test(
  NAME util_math_test
  SRCS "math_test.cpp"
  DEPS sv_util_math sv_gtest_main sv_bench)
cc_bench(
  NAME util_math_bench
  SRCS "math_test.cpp"
  DEPS sv_util_math sv_bench_main sv_gtest)

cc_library(
  NAME util_opencv
  SRCS "opencv.cpp"
  DEPS sv_util_log opencv_core)
cc_test(
  NAME util_opencv_test
  SRCS "opencv_test.cpp"
  DEPS sv_util_opencv sv_gtest_main)

cc_library(
  NAME util_ocvviz
  SRCS "ocvviz.cpp"
  DEPS sv_util_log opencv_core opencv_highgui opencv_imgproc)

cc_library(
  NAME util_imu
  SRCS "imu.cpp"
  DEPS sv_util_math sv_util_eigen Sophus::Sophus)
cc_test(
  NAME util_imu_test
  SRCS "imu_test.cpp"
  DEPS sv_util_imu sv_gtest_main sv_bench)
cc_bench(
  NAME util_imu_bench
  SRCS "imu_test.cpp"
  DEPS sv_util_imu sv_bench_main sv_gtest)

cc_library(
  NAME util_memsize
  SRCS "memsize.cpp"
  DEPS sv_util_log)
cc_test(
  NAME util_memsize_test
  SRCS "memsize_test.cpp"
  DEPS sv_util_memsize sv_gtest_main)

