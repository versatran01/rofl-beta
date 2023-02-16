#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <fstream>

#include "sv/rofl/iviz.h"
#include "sv/rofl/node_util.h"
#include "sv/rofl/odom.h"
#include "sv/util/logging.h"
#include "sv/util/timer.h"

namespace sv::rofl {

namespace nm = nav_msgs;
namespace sm = sensor_msgs;
namespace gm = geometry_msgs;
namespace it = image_transport;
using RowMat34d = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

struct PublishManager {
  PublishManager() = default;
  PublishManager(ros::NodeHandle& pnh, it::ImageTransport& it) {
    pub_sweep_cloud = pnh.advertise<sm::PointCloud2>("sweep/cloud", 1);
    pub_sweep_flat = pnh.advertise<sm::PointCloud2>("sweep/flat", 1);
    pub_grid_points = pnh.advertise<sm::PointCloud2>("grid/points", 1);
    pub_grid_covar = pnh.advertise<vm::MarkerArray>("grid/covar", 1);
    pub_pano_cloud = pnh.advertise<sm::PointCloud2>("pano/cloud", 1);
    pub_pano_covar = pnh.advertise<vm::MarkerArray>("pano/covar", 1);
    pub_pose = pnh.advertise<gm::PoseStamped>("pose", 1);
    pub_traj = pnh.advertise<gm::PoseArray>("traj", 1);
    pub_odom = pnh.advertise<nm::Odometry>("odom", 1);
    pub_path = pnh.advertise<nm::Path>("path", 1);
    pub_pano_poses = pnh.advertise<gm::PoseArray>("pano_poses", 1);
    pub_pano_graph = pnh.advertise<nm::Path>("pano_graph", 1);
    pub_pano = it.advertiseCamera("pano/img", 1);
    pub_pano_viz = it.advertiseCamera("pano/img_viz", 1);
  }

  ros::Publisher pub_sweep_cloud;  // sweep point cloud
  ros::Publisher pub_sweep_flat;   // sweep flat points, alternating colors
  ros::Publisher pub_grid_points;  // grid points
  ros::Publisher pub_grid_covar;   // grid covariance
  ros::Publisher pub_pano_cloud;   // all pano point cloud
  ros::Publisher pub_pano_covar;   // pano covariance
  ros::Publisher pub_traj;         // sweep traj
  ros::Publisher pub_path;         // lidar path
  ros::Publisher pub_odom;         // lidar odom
  ros::Publisher pub_pose;         // lidar pose
  ros::Publisher pub_pano_graph;   // removed panos
  ros::Publisher pub_pano_poses;   // pano in window
  it::CameraPublisher pub_pano;    // pano image and cinfo
  it::CameraPublisher pub_pano_viz;// visualized pano image and cinfo
};

struct FrameManager {
  std::string imu;
  std::string lidar;
  std::string odom{"odom"};    // odom frame starts at identity
  std::string world{"world"};  // world frame gravity aligned
  std::string body{"body"};    // frame for lidar transform tf pub
};

struct NodeOdom {
  /// ros
  ros::NodeHandle pnh_;
  it::ImageTransport it_;
  it::CameraSubscriber sub_scan_;
  ros::Subscriber sub_imu_;
  nav_msgs::Path tum_path_;  // this will not be published

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener sub_tf_;
  tf2_ros::TransformBroadcaster pub_tf_;
  tf2_ros::StaticTransformBroadcaster pub_static_tf_;

  PublishManager pubs_;
  FrameManager frames_;

  int tbb_{0};                  // 0 single thread, >0 tbb grainsize
  int log_{0};                  // 0 no log, >0 log interval
  int vis_{0};                  // 0 disable, 1 sweep, 2 pano, 3 both
  bool est_{true};              // true estimate odom
  bool imu_ready_{false};       // whether imu is ready
  double min_path_dist_{0.01};  // minimum dist before adding pose to path

  LidarOdom odom_;
  Visualizer viz_;

  /// Methods
  explicit NodeOdom(const ros::NodeHandle& pnh);
  void InitRos();
  void InitOdom();
  void Visualize();
  void Publish(const std_msgs::Header& lidar_header);
  void WriteTum() const;

  bool LookupTfImuLidar();
  void SendTfWorldOdom(const Eigen::Vector3d& acc);

  void ImuCb(const sm::Imu& imu_msg);
  void CameraCb(const sm::ImageConstPtr& image_ptr,
                const sm::CameraInfoConstPtr& cinfo_ptr);
};

NodeOdom::NodeOdom(const ros::NodeHandle& pnh)
    : pnh_{pnh}, it_{pnh}, sub_tf_{tf_buffer_} {
  InitRos();
  InitOdom();
}

void NodeOdom::InitRos() {
  // params
  pnh_.getParam("log", log_);
  pnh_.getParam("vis", vis_);
  pnh_.getParam("est", est_);
  ROS_INFO_STREAM(fmt::format("log={}, vis={}, est={}", log_, vis_, est_));

  pnh_.getParam("odom_frame", frames_.odom);
  ROS_INFO_STREAM("odom_frame: " << frames_.odom);
  pnh_.getParam("world_frame", frames_.world);
  ROS_INFO_STREAM("world_frame: " << frames_.world);
  pnh_.getParam("body_frame", frames_.body);
  ROS_INFO_STREAM("body_frame: " << frames_.body);
  pnh_.getParam("min_path_dist", min_path_dist_);
  ROS_INFO_STREAM("min_path_dist: " << min_path_dist_);

  // pub/sub
  const auto ros_hints = ros::TransportHints().tcpNoDelay();
  const auto it_hints = it::TransportHints("raw", ros_hints);
  // queue size is 2 seconds of lidar data, given 10hz full sweep
  sub_scan_ = it_.subscribeCamera(
      "scan/image", 20, &NodeOdom::CameraCb, this, it_hints);
  // queue size is 2 seconds of imu data, given 100hz
  sub_imu_ = pnh_.subscribe("imu", 200, &NodeOdom::ImuCb, this, ros_hints);
  pubs_ = PublishManager(pnh_, it_);

  // send an identity static tf from odom to world on start
  SendTfWorldOdom(Eigen::Vector3d::UnitZ());
}

void NodeOdom::InitOdom() {
  odom_.Init(ReadOdomCfg({pnh_, "odom"}));
  ROS_INFO_STREAM(odom_.cfg().Repr());

  odom_.grid = SweepGrid(ReadGridCfg({pnh_, "grid"}));
  ROS_INFO_STREAM(odom_.grid.Repr());

  odom_.gicp = GicpSolver(ReadGicpCfg({pnh_, "gicp"}));
  ROS_INFO_STREAM(odom_.gicp.Repr());

  odom_.proj = MakeProj({pnh_, "proj"});
  ROS_INFO_STREAM(odom_.proj.Repr());

  odom_.imuq = ImuQueue{ReadImuqCfg({pnh_, "imuq"})};
  ROS_INFO_STREAM(odom_.imuq.Repr());

  odom_.traj = Trajectory{ReadTrajCfg({pnh_, "traj"})};
  ROS_INFO_STREAM(odom_.traj.Repr());

  viz_ = Visualizer(ReadIvizCfg({pnh_, "iviz"}));
  ROS_INFO_STREAM(viz_.cfg().Repr());
}

void NodeOdom::Visualize() {
  if (vis_ & 1) viz_.DrawSweep(odom_.sweep, odom_.grid);
  if (vis_ & 2) viz_.DrawPanos(odom_.pwin, odom_.gicp);
  if (vis_ > 0) viz_.Display();
}

bool NodeOdom::LookupTfImuLidar() {
  if (frames_.imu.empty() || frames_.lidar.empty()) {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        fmt::format("Unable to lookup transform because imu [{}] or lidar [{}] "
                    "frame is empty",
                    frames_.imu,
                    frames_.lidar));
    return false;
  }

  gm::TransformStamped tf_i_l_msg;
  try {
    tf_i_l_msg =
        tf_buffer_.lookupTransform(frames_.imu, frames_.lidar, ros::Time(0));
  } catch (const tf2::TransformException& ex) {
    ROS_ERROR_STREAM(ex.what());
    return false;
  }

  const auto& t = tf_i_l_msg.transform.translation;
  const auto& q = tf_i_l_msg.transform.rotation;
  const Sophus::SE3d tf_i_l{Eigen::Quaterniond{q.w, q.x, q.y, q.z},
                            Eigen::Vector3d{t.x, t.y, t.z}};
  ROS_INFO_STREAM(fmt::format("Transform from lidar [{}] to imu [{}] is\n{}",
                              frames_.imu,
                              frames_.lidar,
                              tf_i_l.matrix3x4()));
  odom_.traj.set_tf_i_l(tf_i_l);
  return true;
}

void NodeOdom::SendTfWorldOdom(const Eigen::Vector3d& acc) {
  // Get rotation from odom to world from body acc and unit z vector.
  // They don't need to be normalized
  // https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html#abb9219bba515452a8ab3f9c38dd45b91
  const auto q_w_o =
      Eigen::Quaterniond::FromTwoVectors(acc, Eigen::Vector3d::UnitZ());

  gm::TransformStamped tf_w_o;
  tf_w_o.header.stamp = ros::Time::now();
  tf_w_o.header.frame_id = frames_.world;
  tf_w_o.child_frame_id = frames_.odom;
  Eigen2Ros(q_w_o, tf_w_o.transform.rotation);
  pub_static_tf_.sendTransform(tf_w_o);
  ROS_INFO_STREAM(fmt::format("Send transform from [{}] to [{}]:\n",
                              tf_w_o.child_frame_id,
                              tf_w_o.header.frame_id)
                  << q_w_o.toRotationMatrix());
}

void NodeOdom::ImuCb(const sm::Imu& imu_msg) {
  static bool tf_ready{false};
  static bool gravity_ready{false};

  static std_msgs::Header prev_header;
  const auto& curr_header = imu_msg.header;

  // Check for dropping message
  if (!prev_header.frame_id.empty()) {
    if (prev_header.seq + 1 != curr_header.seq) {
      ROS_ERROR_STREAM(fmt::format(
          "imu prev seq: {}, curr seq: {}", prev_header.seq, curr_header.seq));
    }
  }
  prev_header = curr_header;

  // Add imu to imuq. This is really the only thing that need to be done on
  // getting an imu msg.
  odom_.imuq.Add(MakeImu(imu_msg));

  // If ready, then we don't need to do any of the following
  if (imu_ready_) return;

  // Set imu frame
  if (frames_.imu.empty()) {
    frames_.imu = curr_header.frame_id;
    ROS_INFO_STREAM(fmt::format("Imu frame: {}", frames_.imu));
  }

  // Lookup transform
  // TODO (rofl): need to actually use this tf
  if (!tf_ready) {
    tf_ready = LookupTfImuLidar();
  }

  // Determine gravity direction
  auto& imuq = odom_.imuq;
  if (!gravity_ready && imuq.full()) {
    const auto acc = imuq.CalcAccMean();
    const auto gyr = imuq.CalcGyrMean();

    ROS_INFO_STREAM(fmt::format("acc mean: {}, norm: {}, using {}/{} imu",
                                acc.mean.transpose(),
                                acc.mean.norm(),
                                acc.n,
                                imuq.size()));

    ROS_INFO_STREAM(fmt::format("gyr mean: {}, norm: {}, using {}/{} imu",
                                gyr.mean.transpose(),
                                gyr.mean.norm(),
                                gyr.n,
                                imuq.size()));

    // Get rotation from unit z and body acc and send as tf_world_odom, such
    // that if we set inerital frame as fixed frame in rviz we get gravity
    // aligned visualization
    if (acc.n > 10) {
      SendTfWorldOdom(acc.mean);
      odom_.traj.set_gravity(acc.mean);
      gravity_ready = true;
    }

    // TODO (rofl): need a better way to set this?
    const auto gyr_mean_norm = gyr.mean.norm();
    if (gyr_mean_norm < 0.01) {
      imuq.bias.gyr = gyr.mean;
      ROS_INFO_STREAM(
          fmt::format(LogColor::kBrightBlue,
                      "Gyr mean norm {:.3f} < 0.01, Set gyr bias: {}",
                      gyr_mean_norm,
                      imuq.bias.gyr.transpose()));
    }
  }

  imu_ready_ = tf_ready && gravity_ready;

  // imu ready implies that we
  // 1. have extrinsic transform from lidar to imu
  // 2. have static transform from odom to inertial
  // 3. have initial gyro bias
  if (imu_ready_) {
    ROS_INFO_STREAM("Imu ready");
  }
}

void NodeOdom::CameraCb(const sm::ImageConstPtr& image_ptr,
                        const sm::CameraInfoConstPtr& cinfo_ptr) {
  static bool scan_ready{false};
  static std_msgs::Header prev_header;
  const auto& curr_header = image_ptr->header;

  // Check for dropping message
  if (!prev_header.frame_id.empty()) {
    if (prev_header.seq + 1 != curr_header.seq) {
      ROS_ERROR_STREAM(fmt::format("lidar prev seq: {}, curr seq: {}",
                                   prev_header.seq,
                                   curr_header.seq));
    }
  }
  prev_header = curr_header;

  // Set lidar frame
  if (frames_.lidar.empty()) {
    frames_.lidar = curr_header.frame_id;
    ROS_INFO_STREAM(fmt::format("Lidar frame: {}", frames_.imu));
  }

  // Allocate odom
  if (odom_.empty()) {
    odom_.Allocate(cv::Size(cinfo_ptr->width, cinfo_ptr->height));
  }

  // We wait for ready flag which is set in ImuCb before doing anything
  // Ready is set when we have 1) extrinsics from lidar to imu and 2) set
  // gravity direction
  if (!imu_ready_) {
    ROS_WARN_STREAM("Waiting for imu ready.");
    return;
  }

  // Wait for the first scan in a sweep, which is stored in cinfo.binning_x
  if (!scan_ready) {
    const auto iscan = cinfo_ptr->binning_x;
    if (iscan == 0) {
      ROS_INFO("Scan ready");
      scan_ready = true;
    } else {
      ROS_WARN_STREAM("Skip scan: " << iscan);
      return;
    }
  }

  // Create new scan
  const auto scan = MakeScan(image_ptr, *cinfo_ptr);
  ROS_DEBUG_STREAM(scan.Repr());
  odom_.AddScan(scan);

  // Only estimate when span is full (this depends on rate multiplier)
  if (odom_.IsSpanFull()) {
    if (est_) {
      odom_.Estimate();
      odom_.UpdateMap();
    }
    Publish(curr_header);

    if (vis_ > 0) Visualize();
    if (log_ > 0) ROS_INFO_STREAM_THROTTLE(log_, odom_.Timings());
  }
}

void NodeOdom::Publish(const std_msgs::Header& lidar_header) {
  // TODO(rofl): only draw what's in odom.span when rate > 1
  const auto& lidar = odom_.proj;
  const auto& sweep = odom_.sweep;
  const auto& grid = odom_.grid;
  const auto& traj = odom_.traj;
  const auto& gicp = odom_.gicp;
  const auto& pwin = odom_.pwin;

  const auto sweep_size = sweep.size2d();
  const auto grid_size = grid.size2d();

  static Cloud2Helper sweep_cloud(sweep_size.height, sweep_size.width, "xyzi");
  static Cloud2Helper sweep_pixels(sweep_size.height, sweep_size.width, "xyzi");
  static Cloud2Helper grid_points(grid_size.height, grid_size.width, "xyzi");
  static Cloud2Helper pwin_cloud(lidar.rows(), lidar.cols(), "xyzi");

  std_msgs::Header odom_header;
  odom_header.frame_id = frames_.odom;
  odom_header.stamp = lidar_header.stamp;

  Timer t;
  t.Start();

  // Publish undistorted sweep in odom frame
  if (pubs_.pub_sweep_cloud.getNumSubscribers() > 0) {
    SweepPoints2Cloud(sweep, sweep_cloud);
    sweep_cloud.cloud.header = odom_header;
    pubs_.pub_sweep_cloud.publish(sweep_cloud.cloud);
  }

  // Publish undistorted selected pixels in odom frame
  if (pubs_.pub_sweep_flat.getNumSubscribers() > 0) {
    SweepPixels2Cloud(sweep, grid, sweep_pixels);
    sweep_pixels.cloud.header = odom_header;
    pubs_.pub_sweep_flat.publish(sweep_pixels.cloud);
  }

  // Publish undistorted grid points in odom frame
  if (pubs_.pub_grid_points.getNumSubscribers() > 0) {
    GridPoints2Cloud(grid, gicp.pinds(), grid_points);
    grid_points.cloud.header = odom_header;
    pubs_.pub_grid_points.publish(grid_points.cloud);
  }

  static vm::MarkerArray grid_covar;
  grid_covar.markers.resize(grid.size());
  // Publish undistorted grid covar in odom frame
  if (pubs_.pub_grid_covar.getNumSubscribers() > 0) {
    GridPoints2Markers(grid, odom_header, grid_covar.markers);
    pubs_.pub_grid_covar.publish(grid_covar);
  }

  // TODO (rofl): This needs to be fixed and put in odom frame
  static vm::MarkerArray pano_covar;
  pano_covar.markers.resize(grid.size());
  if (pubs_.pub_pano_covar.getNumSubscribers() > 0) {
    // Publish pano covar in pano frame
    Gicp2Markers(gicp, pwin, odom_header, pano_covar.markers);
    pubs_.pub_pano_covar.publish(pano_covar);
  }

  if (pubs_.pub_pano_cloud.getNumSubscribers() > 0 && !pwin.empty()) {
    // Publish pano cloud in pano frame
    Pwin2Cloud(pwin, lidar, pwin_cloud);
    pwin_cloud.cloud.header = odom_header;
    pubs_.pub_pano_cloud.publish(pwin_cloud.cloud);
  }

  static gm::PoseArray traj_msg;
  traj_msg.poses.resize(traj.size());

  // Publish traj poses in odom frame
  if (pubs_.pub_traj.getNumSubscribers() > 0) {
    traj_msg.header = odom_header;
    for (size_t i = 0; i < traj.size(); ++i) {
      const auto& st = traj.StateAt(i);
      auto& pose_msg = traj_msg.poses.at(i);
      Eigen2Ros(st.pos, pose_msg.position);
      Sophus2Ros(st.rot, pose_msg.orientation);
    }
    pubs_.pub_traj.publish(traj_msg);
  }

  // Get pose of most recent lidar
  const auto tf_o_l = traj.GetTfOdomLidar();

  // Pose message
  gm::PoseStamped posest_msg;
  posest_msg.header = odom_header;
  Sophus2Ros(tf_o_l, posest_msg.pose);
  pubs_.pub_pose.publish(posest_msg);

  // Pose TF
  gm::TransformStamped posest_tf_msg;
  posest_tf_msg.header = odom_header;
  posest_tf_msg.child_frame_id = frames_.body;
  Sophus2Ros(tf_o_l, posest_tf_msg.transform);
  pub_tf_.sendTransform(posest_tf_msg);

  static nm::Path path_msg;
  path_msg.poses.reserve(1024);

  // Add to tum_path no matter what
  tum_path_.poses.push_back(posest_msg);

  if (pubs_.pub_path.getNumSubscribers() > 0) {
    path_msg.header = odom_header;

    if (path_msg.poses.empty()) {
      path_msg.poses.push_back(posest_msg);
    } else {
      // Compute distance from previous pose
      CHECK(!path_msg.poses.empty());
      const auto& last_pose = path_msg.poses.back();
      Eigen::Map<const Eigen::Vector3d> last_pos(&last_pose.pose.position.x);
      const double dist_sq = (tf_o_l.translation() - last_pos).squaredNorm();

      // only add to path if distance is large enough (this avoids adding too
      // many poses to path)
      if (dist_sq > (min_path_dist_ * min_path_dist_)) {
        path_msg.poses.push_back(posest_msg);
      }
    }

    pubs_.pub_path.publish(path_msg);
  }

  // Publish pano poses
  static gm::PoseArray pano_poses_msg;
  pano_poses_msg.poses.reserve(pwin.capacity());

  if (pubs_.pub_pano_poses.getNumSubscribers() > 0) {
    pano_poses_msg.header = odom_header;
    pano_poses_msg.poses.clear();

    for (int i = 0; i < pwin.size(); ++i) {
      gm::Pose pose_msg;
      Sophus2Ros(pwin.At(i).tf_o_p(), pose_msg);
      pano_poses_msg.poses.push_back(pose_msg);
    }
    pubs_.pub_pano_poses.publish(pano_poses_msg);
  }

  static nm::Path pano_path_msg;
  pano_path_msg.poses.reserve(128);

  static int pano_old_id{0};
  const auto& pano_rm = pwin.removed();
  const auto& pano_new = pwin.first();
  const auto& pano_old = pwin.last();
  static auto cinfo_msg{boost::make_shared<sensor_msgs::CameraInfo>()};

  // At the beginning pano_old.id() should be 0
  if (pano_old.id() > pano_old_id) {
    // Publish removed pano poses as a graph (path)
    if (pubs_.pub_pano_graph.getNumSubscribers() > 0) {
      pano_path_msg.header = odom_header;

      // Removed pano is put into the removed slot in window. If the pano id of
      // that slot changes, then we add this pano's pose to the graph
      gm::PoseStamped pano_pose_msg;
      pano_pose_msg.header.frame_id = frames_.odom;
      pano_pose_msg.header.stamp.fromNSec(pano_rm.time_ns());
      Sophus2Ros(pano_rm.tf_o_p(), pano_pose_msg.pose);
      pano_path_msg.poses.push_back(pano_pose_msg);

      pubs_.pub_pano_graph.publish(pano_path_msg);
    }

    // Publish removed pano as image and cinfo
    if (pubs_.pub_pano.getNumSubscribers() > 0 || 
        pubs_.pub_pano_viz.getNumSubscribers() > 0) 
    {
      cinfo_msg->header = odom_header;
      cinfo_msg->header.stamp.fromNSec(pano_new.time_ns());
      cinfo_msg->width = pano_new.size2d().width;
      cinfo_msg->height = pano_new.size2d().height;
      Eigen::Map<RowMat34d> P_map(&cinfo_msg->P[0]);
      P_map = pano_new.tf_o_p().matrix3x4();
      cinfo_msg->R[0] = PanoData::kRangeScale;
      if (pubs_.pub_pano.getNumSubscribers() > 0) {
        const auto image_msg =
            cv_bridge::CvImage(cinfo_msg->header, "16UC4", pano_new.mat())
                .toImageMsg();
        pubs_.pub_pano.publish(image_msg, cinfo_msg);
      }
      if (pubs_.pub_pano_viz.getNumSubscribers() > 0) {
        // Extract first channel (depth)
        cv::Mat d_channel;
        cv::extractChannel(pano_new.mat(), d_channel, 0);
        
        const auto image_msg =
            cv_bridge::CvImage(cinfo_msg->header, "bgr8", 
                ApplyCmap(d_channel, 1.0 / 65536, cv::COLORMAP_JET)).toImageMsg();
        pubs_.pub_pano_viz.publish(image_msg, cinfo_msg);
      }
    }

    // update pano_old_id;
    pano_old_id = pano_old.id();
  }

  const auto e = t.Elapsed();
  ROS_DEBUG_STREAM(
      fmt::format("Publish using {} ms", static_cast<double>(e) / 1e6));
}

void NodeOdom::WriteTum() const {
  if (tum_path_.poses.empty()) {
    LOG(WARNING) << "Empty path, not writing";
    return;
  }

  const std::string filename("/tmp/tum.txt");
  std::ofstream ofs(filename);

  for (const auto& pose_st : tum_path_.poses) {
    const auto t = pose_st.header.stamp.toSec();
    const auto& p = pose_st.pose.position;
    const auto& q = pose_st.pose.orientation;
    const auto line = fmt::format(
        "{} {} {} {} {} {} {} {}", t, p.x, p.y, p.z, q.x, q.y, q.z, q.w);
    ofs << line << "\n";
  }
  ofs.flush();

  LOG(INFO) << fmt::format(
      "Writing {} tum poses to: {}", tum_path_.poses.size(), filename);
}

}  // namespace sv::rofl

int main(int argc, char** argv) {
  ros::init(argc, argv, "rofl_odom");
  cv::setNumThreads(4);
  sv::rofl::NodeOdom node(ros::NodeHandle("~"));
  ros::spin();

  //node.WriteTum();
  return 0;
}
