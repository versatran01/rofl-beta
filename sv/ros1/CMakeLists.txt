cc_library(
  NAME ros1
  DEPS ${catkin_LIBRARIES}
  INCDIRS ${catkin_INCLUDE_DIRS})

cc_library(
  NAME ros1_msg_conv
  SRCS "msg_conv.cpp"
  DEPS sv_base sv_ros1 sv_util_log Eigen3::Eigen Sophus::Sophus)
