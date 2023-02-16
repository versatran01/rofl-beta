#pragma once

#include <sophus/se3.hpp>

#include "sv/rofl/imuq.h"

namespace sv::rofl {

/// @brief Navigation state, contains time, pos, quat, vel
struct NavState {
  double time{};
  Sophus::SO3d rot{};
  Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel{Eigen::Vector3d::Zero()};

  Sophus::SE3d tf() const noexcept { return {rot, pos}; }
  std::string Repr() const;
};

struct TrajCfg {
  bool use_acc{false};
  bool update_bias{false};
  bool motion_comp{true};

  std::string Repr() const;
};

using IntPair = std::pair<int, int>;
using TimePair = std::pair<double, double>;

/// @brief Trajectory of the sweep buffer at imu time in odom frame
class Trajectory {
  TrajCfg cfg_;
  Sophus::SE3d tf_i_l_;           // transform from lidar to imu
  Eigen::Vector3d gravity_i_;     // gravity in imu frame
  std::vector<NavState> states_;  // state of imu in odom during sweep buffer

 public:
  explicit Trajectory(const TrajCfg& cfg = {}) : cfg_{cfg} {}
  size_t Allocate(int grid_cols);
  std::string Repr() const;

  const auto& cfg() const noexcept { return cfg_; }
  const auto& states() const noexcept { return states_; }
  const Eigen::Vector3d& gravity() const noexcept { return gravity_i_; }
  const Sophus::SE3d& tf_imu_lidar() const noexcept { return tf_i_l_; }

  void set_gravity(const Eigen::Vector3d& gravity_i) { gravity_i_ = gravity_i; }
  void set_tf_i_l(const Sophus::SE3d& tf_i_l) { tf_i_l_ = tf_i_l; }
  /// @brief Get the latest estimate of lidar to odom transform
  Sophus::SE3d GetTfOdomLidar() const;
  Sophus::SO3d GetRotWorldOdom() const;

  double duration() const { return last().time - first().time; }
  size_t size() const noexcept { return states_.size(); }
  int length() const noexcept { return static_cast<int>(states_.size()); }
  int segments() const noexcept { return length() - 1; }
  bool empty() const noexcept { return states_.empty(); }
  bool ok() const { return states_.front().time > 0; }

  NavState& StateAt(int i) { return states_.at(i); }
  const NavState& StateAt(int i) const { return states_.at(i); }

  IntPair PredictLast(const ImuQueue& imuq,
                      const TimePair& time_range,
                      int num_segs);

  /// @brief Predict new states in traj given imu queue
  IntPair PredictNew(const ImuQueue& imuq,
                     const TimePair& time_range,
                     int num_segs);
  IntPair PredictFull(const ImuQueue& imuq);

  /// @brief Pop oldest n states, implemented using std::rotate
  void PopOldest(int n);

  /// @brief Interpolate sweep tfs
  /// @param offset is from the start of the cell
  void Interp(std::vector<Sophus::SE3f>& tfs,
              int col_end,
              double offset = 0,
              int gsize = 0) const;
  void Interp2(std::vector<Sophus::SE3f>& tfs,
               int col_end,
               double offset = 0,
               int gsize = 0) const;

  /// @brief Update first valid traj and return its index
  void UpdateFirst(const Sophus::SE3d& dtf_l);

  const NavState& first() const;
  const NavState& last() const;
};

/// @brief Get delta rotation at time given two imus for dt
Sophus::SO3d DeltaRot(double time,
                      const ImuData& imu0,
                      const ImuData& imu1,
                      double dt);

constexpr double InterpTime(double t, double t0, double t1) {
  return t0 == t1 ? 0.0 : std::clamp((t - t0) / (t1 - t0), 0.0, 1.0);
}

}  // namespace sv::rofl
