#pragma once

#include "sv/rofl/gicp.h"
#include "sv/rofl/grid.h"
#include "sv/rofl/pwin.h"
#include "sv/rofl/traj.h"
#include "sv/util/summary.h"

namespace sv::rofl {

/// @brief Odom Config
struct OdomCfg {
  int tbb{0};                      // tbb grain size
  int rate_factor{1};              // odom rate factor [1, 2, 4, 8,...]
  int num_panos{2};                // num panos in window
  int pano_min_sweeps{10};         // min sweeps for adding new pano
  int pano_max_sweeps{0};          // max sweeps for adding new pano
  double pano_max_trans{5.0};      // max translation to creat a new pano
  double pano_match_ratio{0.8};    // match ratio to create a new pano
  bool use_signal{false};          // use signal data
  bool pano_render_prev{false};    // render previous into new
  bool pano_align_gravity{false};  // use gravity aligned pano

  const OdomCfg& Check() const;
  std::string Repr() const;
};

/// @brief Odom Information
struct OdomInfo {
  int num_selected{};  // num selected points
  int num_matched{};   // num matched points
  bool add_pano{};     // added new pano

  double match_ratio() const {
    return static_cast<double>(num_matched) / num_selected;
  }

  std::string Repr() const;
};

/// @brief Lidar Odometry
class LidarOdom {
  OdomCfg cfg_;
  OdomInfo info_;
  TimerSummary ts_{"rofl"};  // timer summary

  int pano_id_{};               // pano id
  cv::Range buf_span_{};        // tracks sweep buffer span
  size_t total_bytes_{};        // total bytes allocated
  std::string msg_add_pano_{};  // reason for adding pano
  std::string msg_rm_pano_{};   //

 public:
  ImuQueue imuq;     // imu queue
  SweepGrid grid;    // sweep grid
  Trajectory traj;   // sweep trajectory
  GicpSolver gicp;   // generalized icp
  Projection proj;   // projection model
  PanoWindow pwin;   // pano window
  LidarSweep sweep;  // sweep buffer

  size_t Allocate(cv::Size sweep_size);
  void Init(const OdomCfg& cfg) { cfg_ = cfg.Check(); }

  /// @group Info
  const OdomCfg& cfg() const noexcept { return cfg_; }
  const OdomInfo& info() const noexcept { return info_; }

  bool empty() const noexcept { return total_bytes_ == 0; }
  int max_cols() const noexcept { return sweep.cols() / cfg_.rate_factor; }
  bool IsSpanFull() const noexcept { return buf_span_.size() >= max_cols(); }
  bool IsBufferFull() const noexcept { return buf_span_.end >= sweep.cols(); }

  /// @group Odometry functions
  void AddScan(const LidarScan& scan);
  /// @brief Estimate odom
  /// @return true if estimate was ok, false otherwise
  bool Estimate();
  /// @brief Update map
  /// @return true if an old pano was removed
  void UpdateMap();

  /// @group Util
  std::string Timings() const;

 private:
  bool ShouldAddPano(const Sophus::SE3d& tf_o_l);
  void LogOdomAndMapTimes();
};

}  // namespace sv::rofl
