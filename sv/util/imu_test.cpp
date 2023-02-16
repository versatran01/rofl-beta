#include "sv/util/imu.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "imu.h"

namespace sv {
namespace {

using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

TEST(ImuTest, TestImuDataHasNan) {
  ImuData imu{1.0, Vector3d(1, 2, 3), Vector3d(4, 5, 6)};
  EXPECT_FALSE(imu.HasNan());

  imu.acc.x() = kNaND;
  EXPECT_TRUE(imu.HasNan());
}

TEST(ImuTest, TestImuDataDebias) {
  ImuData imu{1.0, Vector3d(1, 2, 3), Vector3d(4, 5, 6)};
  ImuBias bias{Vector3d(1, 1, 1), Vector3d(2, 2, 2)};

  const auto imu_nobias = imu.Debiased(bias);
  EXPECT_EQ(imu_nobias.time, 1.0);
  EXPECT_EQ(imu_nobias.acc, Vector3d(0, 1, 2));
  EXPECT_EQ(imu_nobias.gyr, Vector3d(2, 3, 4));

  imu.Debias(bias);
  EXPECT_EQ(imu.time, 1.0);
  EXPECT_EQ(imu.acc, Vector3d(0, 1, 2));
  EXPECT_EQ(imu.gyr, Vector3d(2, 3, 4));
}

TEST(ImuTest, TestImuPreintUpdate) {
  ImuPreint preint(ImuBias{});
  ImuNoise noise(1.0, 2.0);

  EXPECT_EQ(preint.num_imus, 0);
  EXPECT_EQ(preint.duration, 0);
  EXPECT_EQ(preint.delta.p, Vector3d(0, 0, 0));
  EXPECT_EQ(preint.delta.v, Vector3d(0, 0, 0));

  EXPECT_TRUE((preint.P.array() == 0).all()) << "\n" << preint.P;
  EXPECT_TRUE((preint.J.array() == 0).all()) << "\n" << preint.J;

  // Do some preintegration
  for (int i = 0; i < 10; ++i) {
    preint.Update(0.1, {}, noise);
  }
  EXPECT_EQ(preint.num_imus, 10);
  EXPECT_DOUBLE_EQ(preint.duration, 1.0);

  // J_q_ba should be 0
  EXPECT_EQ((preint.J.bottomLeftCorner<3, 3>()), Matrix3d::Zero());

  preint.Reset();
  EXPECT_EQ(preint.num_imus, 0);
  EXPECT_EQ(preint.duration, 0);
  EXPECT_EQ(preint.delta.p, Vector3d(0, 0, 0));
  EXPECT_EQ(preint.delta.v, Vector3d(0, 0, 0));

  EXPECT_TRUE((preint.P.array() == 0).all()) << "\n" << preint.P;
  EXPECT_TRUE((preint.J.array() == 0).all()) << "\n" << preint.J;
}

/// ============================================================================
namespace bm = benchmark;

void BM_ImuPreintUpdate(bm::State& state) {
  ImuPreint preint(ImuBias{});
  ImuNoise noise(0.0, 0.0);

  for (auto _ : state) {
    preint.Update(0.1, {}, noise);
    bm::DoNotOptimize(preint.delta);
  }
}
BENCHMARK(BM_ImuPreintUpdate);

void BM_ImuPreintBiasCorrect(bm::State& state) {
  ImuBias bias{};
  ImuPreint preint(bias);

  for (auto _ : state) {
    const auto delta = preint.BiasCorrectedDelta(bias);
    bm::DoNotOptimize(delta);
  }
}
BENCHMARK(BM_ImuPreintBiasCorrect);

}  // namespace
}  // namespace sv
