cc_library(
  NAME rofl_proj
  SRCS "proj.cpp"
  DEPS sv_util_log sv_util_math opencv_core)
cc_test(
  NAME rofl_proj_test
  SRCS "proj_test.cpp"
  DEPS sv_rofl_proj sv_gtest_main sv_bench)
cc_bench(
  NAME rofl_proj_bench
  SRCS "proj_test.cpp"
  DEPS sv_rofl_proj sv_bench_main sv_gtest)

cc_library(
  NAME rofl_scan
  SRCS "scan.cpp"
  DEPS sv_util_log sv_util_tbb opencv_core Sophus::Sophus)
cc_test(
  NAME rofl_scan_test
  SRCS "scan_test.cpp"
  DEPS sv_rofl_scan sv_gtest_main sv_bench)
cc_bench(
  NAME rofl_scan_bench
  SRCS "scan_test.cpp"
  DEPS sv_rofl_scan sv_bench_main sv_gtest)

cc_library(
  NAME rofl_grid
  SRCS "grid.cpp"
  DEPS sv_rofl_scan sv_util_eigen sv_util_grid2d)
cc_test(
  NAME rofl_grid_test
  SRCS "grid_test.cpp"
  DEPS sv_rofl_grid sv_gtest_main sv_bench)
cc_bench(
  NAME rofl_grid_bench
  SRCS "grid_test.cpp"
  DEPS sv_rofl_grid sv_bench_main sv_gtest)

cc_library(
  NAME rofl_pano
  SRCS "pano.cpp"
  DEPS sv_rofl_proj sv_rofl_scan)
cc_test(
  NAME rofl_pano_test
  SRCS "pano_test.cpp"
  DEPS sv_rofl_pano sv_gtest_main sv_bench)
cc_bench(
  NAME rofl_pano_bench
  SRCS "pano_test.cpp"
  DEPS sv_rofl_pano sv_bench_main sv_gtest)

cc_library(
  NAME rofl_imuq
  SRCS "imuq.cpp"
  DEPS sv_util_imu sv_util_log)
cc_test(
  NAME rofl_imuq_test
  SRCS "imuq_test.cpp"
  DEPS sv_rofl_imuq sv_gtest_main sv_bench)

cc_library(
  NAME rofl_traj
  SRCS "traj.cpp"
  DEPS sv_rofl_imuq sv_util_tbb Sophus::Sophus)
cc_test(
  NAME rofl_traj_test
  SRCS "traj_test.cpp"
  DEPS sv_rofl_traj sv_gtest_main sv_bench)
cc_bench(
  NAME rofl_traj_bench
  SRCS "traj_test.cpp"
  DEPS sv_rofl_traj sv_bench_main sv_gtest)

cc_library(
  NAME rofl_hess
  SRCS "hess.cpp"
  DEPS sv_util_eigen Sophus::Sophus)
cc_test(
  NAME rofl_hess_test
  SRCS "hess_test.cpp"
  DEPS sv_rofl_hess sv_gtest_main sv_bench)
cc_bench(
  NAME rofl_hess_bench
  SRCS "hess_test.cpp"
  DEPS sv_rofl_hess sv_bench_main sv_gtest)

cc_library(
  NAME rofl_pwin
  SRCS "pwin.cpp"
  DEPS sv_rofl_pano absl::inlined_vector)
cc_test(
  NAME rofl_pwin_test
  SRCS "pwin_test.cpp"
  DEPS sv_rofl_pwin sv_gtest_main)

cc_library(
  NAME rofl_gicp
  SRCS "gicp.cpp"
  DEPS sv_rofl_grid sv_rofl_traj sv_rofl_hess sv_rofl_pwin)
cc_test(
  NAME rofl_gicp_test
  SRCS "gicp_test.cpp"
  DEPS sv_rofl_gicp sv_gtest_main sv_bench)
cc_bench(
  NAME rofl_gicp_bench
  SRCS "gicp_test.cpp"
  DEPS sv_rofl_gicp sv_bench_main sv_gtest)

cc_library(
  NAME rofl_iviz
  SRCS "iviz.cpp"
  DEPS sv_rofl_pwin sv_rofl_gicp sv_util_ocvviz)

cc_library(
  NAME rofl_odom
  SRCS "odom.cpp"
  DEPS sv_rofl_gicp sv_util_summary sv_util_memsize)

if(BUILD_SV_PLAY)
  cc_binary(
    NAME rofl_gicp_play
    SRCS "gicp_play.cpp"
    DEPS sv_rofl_gicp sv_util_summary absl::flags_parse)
endif()

# ros
if(BUILD_SV_ROS)
  cc_library(
    NAME rofl_node_util
    SRCS "node_util.cpp"
    DEPS sv_ros1_msg_conv sv_rofl_odom)

  cc_binary(
    NAME rofl_node_odom
    SRCS "node_odom.cpp"
    DEPS sv_rofl_node_util sv_rofl_iviz)
endif()
