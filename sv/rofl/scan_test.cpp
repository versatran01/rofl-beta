#include "sv/rofl/scan.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

TEST(Scan, WrapCols) {
  constexpr int cols = 10;
  // [-10, 0)

  EXPECT_EQ(WrapCols(-10, cols), 0);
  EXPECT_EQ(WrapCols(-9, cols), 1);
  EXPECT_EQ(WrapCols(-1, cols), 9);

  // [0, 10)
  EXPECT_EQ(WrapCols(0, cols), 0);
  EXPECT_EQ(WrapCols(1, cols), 1);
  EXPECT_EQ(WrapCols(9, cols), 9);

  // [10, 20)
  EXPECT_EQ(WrapCols(10, cols), 0);
  EXPECT_EQ(WrapCols(11, cols), 1);
  EXPECT_EQ(WrapCols(19, cols), 9);
}

TEST(Scan, ScanData) {
  ScanData data;
  EXPECT_EQ(data.ok(), true);
  EXPECT_EQ(data.xyz(), Eigen::Vector3f::Zero());
}

TEST(Scan, MatBase) {
  const cv::Mat mat = cv::Mat::zeros(10, 20, CV_32FC4);
  MatBase<ScanData> mb(mat);
  EXPECT_EQ(mb.rows(), 10);
  EXPECT_EQ(mb.cols(), 20);

  const auto n = mb.AllocateMat({20, 30});
  EXPECT_EQ(n, 600 * sizeof(ScanData));
}

TEST(Scan, LidarScan) {
  const auto scan = MakeTestScan({1024, 64});
  EXPECT_EQ(scan.rows(), 64);
  EXPECT_EQ(scan.cols(), 1024);
  EXPECT_EQ(scan.type(), ScanData::kDtype);
  EXPECT_EQ(scan.info().range_scale, 512.0F);
}

/// ============================================================================
namespace bm = benchmark;

void BM_CountNumValid(bm::State& state) {
  const auto scan = MakeTestScan({1024, 64});
  const auto gsize = static_cast<int>(state.range(0));

  for (auto _ : state) {
    const auto n = scan.CountNumValid(gsize);
    bm::DoNotOptimize(n);
  }
}
BENCHMARK(BM_CountNumValid)->Arg(0)->Arg(1);

void BM_SweepFillHoles(bm::State& state) {
  auto sweep = MakeTestSweep({1024, 64});
  const auto gsize = static_cast<int>(state.range(0));

  for (auto _ : state) {
    sweep.FillHoles(gsize);
    bm::DoNotOptimize(sweep);
  }
}
BENCHMARK(BM_SweepFillHoles)->Arg(0)->Arg(1);

void BM_SweepCalcRangeGrad2(bm::State& state) {
  auto sweep = MakeTestSweep({1024, 64});
  const auto gsize = static_cast<int>(state.range(0));

  for (auto _ : state) {
    sweep.CalcRangeGrad2(gsize);
    bm::DoNotOptimize(sweep);
  }
}
BENCHMARK(BM_SweepCalcRangeGrad2)->Arg(0)->Arg(1);

void BM_SweepCalcSignalGrad(bm::State& state) {
  auto sweep = MakeTestSweep({1024, 64});
  const auto gsize = static_cast<int>(state.range(0));

  for (auto _ : state) {
    sweep.CalcSignalGrad(gsize);
    bm::DoNotOptimize(sweep);
  }
}
BENCHMARK(BM_SweepCalcSignalGrad)->Arg(0)->Arg(1);

// void BM_SweepCalcSignalEdge(bm::State& state) {
//   auto sweep = MakeTestSweep({1024, 64});
//   const auto gsize = static_cast<int>(state.range(0));
//   sweep.CalcSignalGrad(gsize);

//  for (auto _ : state) {
//    sweep.CalcSignalEdge(50, gsize);
//    bm::DoNotOptimize(sweep);
//  }
//}
// BENCHMARK(BM_SweepCalcSignalEdge)->Arg(0)->Arg(1);

}  // namespace
}  // namespace sv::rofl
