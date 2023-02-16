#include "sv/rofl/imuq.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

TEST(Imu, ImuQueue) {
  ImuqCfg cfg;
  cfg.acc_sigma = cfg.gyr_sigma = cfg.acc_bias_sigma = cfg.gyr_bias_sigma = 1;

  ImuQueue imuq(cfg);
  EXPECT_EQ(imuq.size(), 0);
  EXPECT_EQ(imuq.full(), false);
  EXPECT_EQ(imuq.empty(), true);
  EXPECT_EQ(imuq.capacity(), cfg.bufsize);

  ImuData d;
  d.time = 1;
  EXPECT_EQ(imuq.Add(d), true);
  EXPECT_EQ(imuq.size(), 1);

  d.acc.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(imuq.Add(d), false);
  EXPECT_EQ(imuq.size(), 1);
}

TEST(Imu, ImuIndexAfterTime) {
  ImuBuffer buffer(5);
  // When empty returns buf.size which is also 0
  EXPECT_EQ(ImuIndexAfterTime(buffer, 0), 0);

  for (int i = 0; i < 5; ++i) {
    ImuData d;
    d.time = i + 1;
    buffer.push_back(d);
  }

  // imu time 1, 2, 3, 4, 5
  EXPECT_EQ(ImuIndexAfterTime(buffer, 0), 0);
  EXPECT_EQ(ImuIndexAfterTime(buffer, 0.5), 0);
  EXPECT_EQ(ImuIndexAfterTime(buffer, 1), 1);
  EXPECT_EQ(ImuIndexAfterTime(buffer, 1.5), 1);
  EXPECT_EQ(ImuIndexAfterTime(buffer, 2), 2);
  EXPECT_EQ(ImuIndexAfterTime(buffer, 15), 5);
}

TEST(Imu, ImuIndexBeforeTime) {
  ImuBuffer buffer(5);
  // When empty returns -1
  EXPECT_EQ(ImuIndexBeforeTime(buffer, 0), -1);

  for (int i = 0; i < 5; ++i) {
    ImuData d;
    d.time = i + 1;
    buffer.push_back(d);
  }

  // imu time 1, 2, 3, 4, 5
  EXPECT_EQ(ImuIndexBeforeTime(buffer, 0), -1);
  EXPECT_EQ(ImuIndexBeforeTime(buffer, 0.5), -1);
  EXPECT_EQ(ImuIndexBeforeTime(buffer, 1), 0);
  EXPECT_EQ(ImuIndexBeforeTime(buffer, 1.5), 0);
  EXPECT_EQ(ImuIndexBeforeTime(buffer, 2), 1);
  EXPECT_EQ(ImuIndexBeforeTime(buffer, 15), 4);
}

}  // namespace
}  // namespace sv::rofl
