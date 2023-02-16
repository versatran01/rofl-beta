#pragma once

#include <sophus/so3.hpp>

#include "sv/util/math.h"

namespace sv {

/// @brief Imu bias
struct ImuBias {
  Eigen::Vector3d acc{0, 0, 0};
  Eigen::Vector3d gyr{0, 0, 0};
};

/// @brief Time-stamped Imu data
struct ImuData {
  double time{};
  Eigen::Vector3d acc{0, 0, 0};
  Eigen::Vector3d gyr{0, 0, 0};

  // Whether imu data is bad
  bool HasNan() const {
    return acc.array().isNaN().any() || gyr.array().isNaN().any();
  }

  // Remove bias inplace
  ImuData& Debias(const ImuBias& bias) {
    acc -= bias.acc;
    gyr -= bias.gyr;
    return *this;
  }

  // Returns a new imu data without bias removed
  ImuData Debiased(const ImuBias& bias) const {
    return {time, acc - bias.acc, gyr - bias.gyr};
  }
};

/// @brief Discrete time IMU noise
struct ImuNoise {
  Eigen::Vector3d acc_var{0, 0, 0};
  Eigen::Vector3d gyr_var{0, 0, 0};
  Eigen::Vector3d bias_acc_var{0, 0, 0};
  Eigen::Vector3d bias_gyr_var{0, 0, 0};

  ImuNoise() = default;
  ImuNoise(double acc_sigma,
           double gyr_sigma,
           double bias_acc_sigma = 0.0,
           double bias_gyr_sigma = 0.0) {
    acc_var.setConstant(Sq(acc_sigma));
    gyr_var.setConstant(Sq(gyr_sigma));
    bias_acc_var.setConstant(Sq(bias_acc_sigma));
    bias_gyr_var.setConstant(Sq(bias_gyr_sigma));
  }
};

struct ImuPreint {
  static constexpr int kDimPvq = 9;
  static constexpr int kDimBias = 6;
  static constexpr int kDimFull = kDimPvq + kDimBias;

  using Matrix9d = Eigen::Matrix<double, 9, 9>;
  using Matrix96d = Eigen::Matrix<double, 9, 6>;

  struct Delta {
    Eigen::Vector3d p{0, 0, 0};
    Eigen::Vector3d v{0, 0, 0};
    Sophus::SO3d q{};
  };

  int num_imus{0};       // number of imus used
  double duration{0.0};  // total time integrated

  Delta delta{};
  ImuBias bias_hat{};
  Matrix9d P{Matrix9d::Zero()};
  Matrix96d J{Matrix96d::Zero()};

  // Ctors
  ImuPreint() = default;
  explicit ImuPreint(const ImuBias& bias) : bias_hat(bias) {}

  // Update (pre-integrated) IMU pre-integrated delta given imu data
  void Update(double dt, const ImuData& imu, const ImuNoise& noise);

  void Reset();

  // Given new bias, return first-order bias-corrected IMU pre-integrated deltas
  Delta BiasCorrectedDelta(const ImuBias& bias_new) const;

  // Get the information matrix of the preintegrated delta
  Matrix9d GetInfoPvq() const;
};

}  // namespace sv
