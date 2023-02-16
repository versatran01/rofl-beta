#pragma once

#include <ros/node_handle.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "sv/rofl/iviz.h"
#include "sv/rofl/odom.h"
#include "sv/rofl/scan.h"
#include "sv/ros1/msg_conv.h"

namespace sv::rofl {

namespace vm = visualization_msgs;

/// @brief Read cfg
OdomCfg ReadOdomCfg(const ros::NodeHandle& pnh);
GridCfg ReadGridCfg(const ros::NodeHandle& pnh);
PanoCfg ReadPanoCfg(const ros::NodeHandle& pnh);
GicpCfg ReadGicpCfg(const ros::NodeHandle& pnh);
ImuqCfg ReadImuqCfg(const ros::NodeHandle& pnh);
TrajCfg ReadTrajCfg(const ros::NodeHandle& pnh);
IvizCfg ReadIvizCfg(const ros::NodeHandle& pnh);

/// @brief Make
LidarScan MakeScan(const sensor_msgs::ImageConstPtr& image_msg,
                   const sensor_msgs::CameraInfo& cinfo_msg);
Projection MakeProj(const ros::NodeHandle& pnh);
ImuData MakeImu(const sensor_msgs::Imu& imu_msg);

/// @brief Visualization related functions, tbb by default
void SweepPoints2Cloud(const LidarSweep& sweep, Cloud2Helper& cloud);
void SweepPixels2Cloud(const LidarSweep& sweep,
                       const SweepGrid& grid,
                       Cloud2Helper& cloud);

void GridPoints2Cloud(const SweepGrid& grid,
                      const cv::Mat& pinds,
                      Cloud2Helper& cloud);
void GridPoints2Markers(const SweepGrid& grid,
                        const std_msgs::Header& header,
                        std::vector<vm::Marker>& markers);

void Pano2Cloud(const DepthPano& pano,
                const Projection& proj,
                Cloud2Helper& cloud);
void Pwin2Cloud(const PanoWindow& pwin,
                const Projection& proj,
                Cloud2Helper& cloud);
void Gicp2Markers(const GicpSolver& gicp,
                  const PanoWindow& pwin,
                  const std_msgs::Header& header,
                  std::vector<vm::Marker>& markers);

void Covar2Marker(const MeanCovar3f& mc, vm::Marker& marker, double eps = 1e-8);

MeanCovar3f TransformCovar(MeanCovar3f mc, const Sophus::SE3f& tf);

}  // namespace sv::rofl
