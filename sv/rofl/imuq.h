#pragma once

#include <boost/circular_buffer.hpp>

#include "sv/util/imu.h"

namespace sv::rofl {

using ImuBuffer = boost::circular_buffer<ImuData>;

struct ImuqCfg {
  int bufsize{30};
  double rate{100.0};
  double acc_sigma{};
  double gyr_sigma{};
  double acc_bias_sigma{};
  double gyr_bias_sigma{};

  void Check() const;
  std::string Repr() const;
};

struct ImuQueue {
  explicit ImuQueue(const ImuqCfg& cfg = {});

  ImuBias bias;
  ImuNoise noise;
  ImuBuffer buf{};

  std::string Repr() const;

  auto size() const noexcept { return buf.size(); }
  auto full() const noexcept { return buf.full(); }
  auto empty() const noexcept { return buf.empty(); }
  auto capacity() const noexcept { return buf.capacity(); }

  const ImuData& At(int i) const { return buf.at(i); }
  ImuData DebiasedAt(int i) const { return At(i).Debiased(bias); }
  const ImuData& first() const { return buf.front(); }
  const ImuData& last() const { return buf.back(); }

  /// @brief Add imu data into buffer
  bool Add(const ImuData& imu);

  /// @brief Find imu right after time
  int FindAfter(double t) const;
  int FindBefore(double t) const;

  /// @brief Compute mean acc in buf that is close enough to g
  MeanCovar3d CalcAccMean() const;
  MeanCovar3d CalcGyrMean() const;
};

/// @brief Find imu in buffer that is right after t
/// @return buf.size() if all imu is before t
int ImuIndexAfterTime(const ImuBuffer& buf, double t);

/// @brief Find imu in buffer that is right before t
/// @return -1 if all imu is after t
int ImuIndexBeforeTime(const ImuBuffer& buf, double t);

/// @brief Make test ImuQueue, time starts from 1 to size
ImuQueue MakeTestImuq(int size);

}  // namespace sv::rofl
