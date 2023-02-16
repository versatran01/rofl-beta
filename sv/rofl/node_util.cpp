#include "sv/rofl/node_util.h"

#include <cv_bridge/cv_bridge.h>

#include <Eigen/Eigenvalues>

#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::rofl {

OdomCfg ReadOdomCfg(const ros::NodeHandle& pnh) {
  OdomCfg cfg;

  pnh.getParam("tbb", cfg.tbb);
  pnh.getParam("num_panos", cfg.num_panos);
  pnh.getParam("use_signal", cfg.use_signal);
  pnh.getParam("rate_multiplier", cfg.rate_factor);
  pnh.getParam("pano_max_trans", cfg.pano_max_trans);
  pnh.getParam("pano_min_sweeps", cfg.pano_min_sweeps);
  pnh.getParam("pano_max_sweeps", cfg.pano_max_sweeps);
  pnh.getParam("pano_match_ratio", cfg.pano_match_ratio);
  pnh.getParam("pano_render_prev", cfg.pano_render_prev);
  pnh.getParam("pano_align_gravity", cfg.pano_align_gravity);

  return cfg;
}

GridCfg ReadGridCfg(const ros::NodeHandle& pnh) {
  GridCfg cfg;

  pnh.getParam("cell_rows", cfg.cell_rows);
  pnh.getParam("cell_cols", cfg.cell_cols);
  pnh.getParam("feat_max_smooth", cfg.feat_max_smooth);
  pnh.getParam("feat_min_pixels", cfg.feat_min_pixels);
  pnh.getParam("feat_min_range", cfg.feat_min_range);
  pnh.getParam("feat_min_length", cfg.feat_min_length);
  pnh.getParam("feat_nms_dist", cfg.feat_nms_dist);

  return cfg;
}

PanoCfg ReadPanoCfg(const ros::NodeHandle& pnh) {
  PanoCfg cfg;

  pnh.getParam("max_info", cfg.max_info);
  pnh.getParam("min_range", cfg.min_range);
  pnh.getParam("max_range", cfg.max_range);
  pnh.getParam("fuse_rel_tol", cfg.fuse_rel_tol);
  pnh.getParam("fuse_abs_tol", cfg.fuse_abs_tol);

  return cfg;
}

GicpCfg ReadGicpCfg(const ros::NodeHandle& pnh) {
  GicpCfg cfg;

  pnh.getParam("max_inner_iters", cfg.max_inner_iters);
  pnh.getParam("max_outer_iters", cfg.max_outer_iters);
  pnh.getParam("match_half_rows", cfg.match_half_rows);
  pnh.getParam("match_half_cols", cfg.match_half_cols);
  pnh.getParam("stop_pos_tol", cfg.stop_pos_tol);
  pnh.getParam("stop_rot_tol", cfg.stop_rot_tol);
  pnh.getParam("use_all_panos", cfg.use_all_panos);

  return cfg;
}

ImuqCfg ReadImuqCfg(const ros::NodeHandle& pnh) {
  ImuqCfg cfg;

  pnh.getParam("rate", cfg.rate);
  pnh.getParam("bufsize", cfg.bufsize);
  pnh.getParam("acc_sigma", cfg.acc_sigma);
  pnh.getParam("gyr_sigma", cfg.gyr_sigma);
  pnh.getParam("acc_bias_sigma", cfg.acc_bias_sigma);
  pnh.getParam("gyr_bias_sigma", cfg.gyr_bias_sigma);

  return cfg;
}

TrajCfg ReadTrajCfg(const ros::NodeHandle& pnh) {
  TrajCfg cfg;
  pnh.getParam("use_acc", cfg.use_acc);
  pnh.getParam("update_bias", cfg.update_bias);
  pnh.getParam("motion_comp", cfg.motion_comp);
  return cfg;
}

IvizCfg ReadIvizCfg(const ros::NodeHandle& pnh) {
  IvizCfg cfg;

  pnh.getParam("min_range", cfg.min_range);
  pnh.getParam("max_signal", cfg.max_signal);
  pnh.getParam("disp_scale_pano", cfg.disp_scale_pano);
  pnh.getParam("disp_scale_sweep", cfg.disp_scale_sweep);
  pnh.getParam("screen_width", cfg.screen_width);
  pnh.getParam("screen_height", cfg.screen_height);
  pnh.getParam("show_pano_range", cfg.show_pano_range);
  pnh.getParam("show_pano_signal", cfg.show_pano_signal);
  pnh.getParam("show_pano_info", cfg.show_pano_info);
  pnh.getParam("show_pano_grad", cfg.show_pano_grad);

  return cfg;
}

Projection MakeProj(const ros::NodeHandle& pnh) {
  const auto rows = pnh.param<int>("pano_rows", 256);
  const auto cols = pnh.param<int>("pano_cols", 1024);
  const auto vfov = pnh.param<double>("pano_vfov", 0.0);
  return Projection{{cols, rows}, vfov};
}

LidarScan MakeScan(const sensor_msgs::ImageConstPtr& image_ptr,
                   const sensor_msgs::CameraInfo& cinfo_msg) {
  const auto cv_ptr = cv_bridge::toCvShare(image_ptr);
  const auto& header = image_ptr->header;
  const auto& roi = cinfo_msg.roi;

  ScanInfo info;
  info.col_span.start = roi.x_offset;
  info.col_span.end = info.col_span.start + roi.width;
  info.end_time = header.stamp.toSec();
  info.col_dtime = cinfo_msg.K[0];
  info.range_scale = static_cast<float>(cinfo_msg.R[0]);

  return {cv_ptr->image, info};
}

ImuData MakeImu(const sensor_msgs::Imu& imu_msg) {
  ImuData imu;
  imu.time = imu_msg.header.stamp.toSec();
  const auto& a = imu_msg.linear_acceleration;
  const auto& w = imu_msg.angular_velocity;
  imu.acc = {a.x, a.y, a.z};
  imu.gyr = {w.x, w.y, w.z};
  return imu;
}

void SweepPoints2Cloud(const LidarSweep& sweep, Cloud2Helper& cloud) {
  CHECK_EQ(sweep.rows(), cloud.rows());
  CHECK_EQ(sweep.cols(), cloud.cols());
  CHECK_EQ(cloud.cloud.point_step, 16);

  ParallelFor({0, sweep.rows(), 1}, [&](int sr) {
    for (int sc = 0; sc < sweep.cols(); ++sc) {
      const auto& data_s = sweep.DataAt(sr, sc);
      const auto& tf_o_s = sweep.TfAt(sc);

      auto* ptr = cloud.PtrAt<float>(sr, sc);
      Eigen::Map<Eigen::Vector4f> xyzs(ptr);

      // transform from sweep to odom frame
      xyzs.head<3>() = tf_o_s * data_s.xyz();
      xyzs[3] = data_s.s16u;
    }
  });
}

void SweepPixels2Cloud(const LidarSweep& sweep,
                       const SweepGrid& grid,
                       Cloud2Helper& cloud) {
  CHECK_EQ(sweep.rows(), cloud.rows());
  CHECK_EQ(sweep.cols(), cloud.cols());
  CHECK_EQ(cloud.point_step(), 4 * 4);

  // First fill cloud with points
  ParallelFor({0, sweep.rows(), 1}, [&](int sr) {
    for (int sc = 0; sc < sweep.cols(); ++sc) {
      const auto& data_s = sweep.DataAt(sr, sc);
      const auto& tf_o_s = sweep.TfAt(sc);

      auto* ptr = cloud.PtrAt<float>(sr, sc);
      Eigen::Map<Eigen::Vector4f> xyzs(ptr);

      // transform from sweep to odom frame
      xyzs.head<3>() = tf_o_s * data_s.xyz();
      xyzs[3] = 0;  // intensity channel represent cell
    }
  });

  // Then set valid pixels using alternating color
  const auto& points = grid.points();
  ParallelFor({0, grid.rows(), 1}, [&](int gr) {
    for (int gc = 0; gc < grid.cols(); ++gc) {
      const auto& point = points.at(gr, gc);
      for (int cc = 0; cc < point.width(); ++cc) {
        auto* ptr = cloud.PtrAt<float>(point.xyw.y, point.xyw.x + cc);
        ptr[3] = static_cast<float>(gc % 2 + 1);  // alternating color
      }
    }
  });
}

void GridPoints2Cloud(const SweepGrid& grid,
                      const cv::Mat& pinds,
                      Cloud2Helper& cloud) {
  CHECK_EQ(grid.rows(), cloud.rows());
  CHECK_EQ(grid.cols(), cloud.cols());
  CHECK_EQ(grid.rows(), pinds.rows);
  CHECK_EQ(grid.cols(), pinds.cols);
  CHECK_EQ(pinds.type(), CV_8UC1);
  CHECK_EQ(cloud.point_step(), 4 * 4);

  const auto& points = grid.points();

  ParallelFor({0, points.rows(), 1}, [&](int gr) {
    for (int gc = 0; gc < points.cols(); ++gc) {
      const auto& point = points.at(gr, gc);
      const auto pind = pinds.at<uchar>(gr, gc);

      auto* ptr = cloud.PtrAt<float>(gr, gc);
      Eigen::Map<Eigen::Vector3f> pc(ptr);
      ptr[3] = static_cast<float>(pind);

      if (point.ok()) {
        // set point nan
        pc = grid.TfAt(gc) * point.mc.mean;
      } else {
        pc.setConstant(kNaNF);
      }
    }
  });
}

void Pano2Cloud(const DepthPano& pano,
                const Projection& proj,
                Cloud2Helper& cloud) {
  CHECK_LE(pano.rows(), cloud.rows());
  CHECK_EQ(pano.cols(), cloud.cols());
  CHECK_EQ(cloud.point_step(), 4 * 4);

  // We only need pano points in local frame, since the node will publish a tf
  // from pano to odom frame
  ParallelFor({0, pano.rows(), 1}, [&](int r) {
    for (int c = 0; c < pano.cols(); ++c) {
      const auto& data = pano.DataAt(r, c);
      auto* ptr = cloud.PtrAt<float>(r, c);

      if (data.bad()) {
        ptr[0] = ptr[1] = ptr[2] = kNaNF;
        continue;
      }
      const auto rg = data.GetRange();
      const auto pt = proj.Backward(r, c, rg);
      ptr[0] = static_cast<float>(pt.x);
      ptr[1] = static_cast<float>(pt.y);
      ptr[2] = static_cast<float>(pt.z);
      ptr[3] = static_cast<float>(data.s16u);
    }
  });  // row
}

void Pwin2Cloud(const PanoWindow& pwin,
                const Projection& proj,
                Cloud2Helper& cloud) {
  CHECK(!pwin.empty());
  cloud.resize(pwin.size() * proj.rows(), proj.cols());

  CHECK_EQ(cloud.point_step(), 4 * 4);

  ParallelFor({0, pwin.size(), 1}, [&](int i) {
    const auto& pano = pwin.At(i);
    const auto tf = pano.tf_o_p().cast<float>();

    ParallelFor({0, proj.rows(), 1}, [&](int r) {
      for (int c = 0; c < proj.cols(); ++c) {
        auto* ptr = cloud.PtrAt<float>(r + i * pano.rows(), c);
        Eigen::Map<Eigen::Vector3f> xyz(ptr);
        const auto& data = pano.DataAt(r, c);

        if (data.bad()) {
          xyz.setConstant(kNaNF);
          continue;
        }

        const auto rg = data.GetRange();
        const auto pt = proj.Backward(r, c, rg);

        xyz.x() = static_cast<float>(pt.x);
        xyz.y() = static_cast<float>(pt.y);
        xyz.z() = static_cast<float>(pt.z);

        // Transform to odom frame
        xyz = tf * xyz;
        ptr[3] = static_cast<float>(data.s16u);
      }
    });  // row
  });    // pano
}

void Covar2Marker(const MeanCovar3f& mc, vm::Marker& marker, double eps) {
  CHECK(mc.ok());

  // Add small value along diagonal to avoid numerical issues
  auto covar = mc.Covar();
  if (eps > 0) covar.diagonal().array() += static_cast<float>(eps);

  // Compute eigenvalues and eigenvector, then make sure it is right-handed
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covar);
  auto eigvals = es.eigenvalues().cast<double>().eval();
  auto eigvecs = es.eigenvectors().cast<double>().eval();
  MakeRightHanded(eigvals, eigvecs);

  Eigen2Ros(mc.mean.cast<double>(), marker.pose.position);
  Eigen2Ros(Eigen::Quaterniond(eigvecs), marker.pose.orientation);
  Eigen2Ros(eigvals.cwiseSqrt() * 2, marker.scale);
}

void GridPoints2Markers(const SweepGrid& grid,
                        const std_msgs::Header& header,
                        std::vector<vm::Marker>& markers) {
  CHECK_EQ(grid.size(), markers.size());

  const auto& points = grid.points();

  ParallelFor({0, points.rows(), 1}, [&](int gr) {
    for (int gc = 0; gc < points.cols(); ++gc) {
      const int i = points.rc2ind(gr, gc);
      const auto& point = points.at(i);
      auto& marker = markers.at(i);

      marker.ns = "grid";
      marker.id = i;

      if (!point.mc.ok()) {
        marker.action = vm::Marker::DELETE;
        continue;
      }

      marker.color.a = 0.8F;
      marker.color.r = 1.0F;
      marker.header = header;
      marker.type = vm::Marker::SPHERE;
      marker.action = vm::Marker::ADD;

      const auto& tf = grid.TfAt(gc);
      Covar2Marker(TransformCovar(point.mc, tf), marker);
    }
  });
}

void Gicp2Markers(const GicpSolver& gicp,
                  const PanoWindow& pwin,
                  const std_msgs::Header& header,
                  std::vector<vm::Marker>& markers) {
  CHECK_EQ(gicp.matches().size(), markers.size());

  const auto& matches = gicp.matches();
  const auto& pinds = gicp.pinds();
  const auto npanos = pwin.size();

  static SE3fVec tfs_o_p;
  tfs_o_p.clear();
  tfs_o_p.reserve(npanos);
  for (int i = 0; i < npanos; ++i) {
    tfs_o_p.push_back(pwin.At(i).tf_o_p().cast<float>());
  }

  ParallelFor({0, matches.rows(), 1}, [&](int gr) {
    for (int gc = 0; gc < matches.cols(); ++gc) {
      const int i = matches.rc2ind(gr, gc);
      const auto& match = matches.at(i);
      auto& marker = markers.at(i);

      marker.ns = "pano";
      marker.id = i;

      const auto pind = static_cast<int>(pinds.at<uchar>(gr, gc));
      if (match.bad() || pind >= npanos) {
        marker.action = vm::Marker::DELETE;
        continue;
      }

      marker.color.a = 0.5;
      marker.color.g = 1.0;
      marker.header = header;
      marker.type = vm::Marker::SPHERE;
      marker.action = vm::Marker::ADD;

      const auto& tf = tfs_o_p.at(pind);
      Covar2Marker(TransformCovar(match.mc, tf), marker);
    }
  });
}

MeanCovar3f TransformCovar(MeanCovar3f mc, const Sophus::SE3f& tf) {
  mc.mean = tf * mc.mean;
  const auto R = tf.rotationMatrix();
  mc.covar_sum = R * mc.covar_sum * R.transpose();
  return mc;
}

}  // namespace sv::rofl
