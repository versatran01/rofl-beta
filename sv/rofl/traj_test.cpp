#include "sv/rofl/traj.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

TEST(Traj, PredictNew) {
  const auto imuq = MakeTestImuq(10);

  TrajCfg cfg;
  Trajectory traj(cfg);
  traj.Allocate(8);

  const auto p = traj.PredictNew(imuq, {1, 5}, 4);
  EXPECT_EQ(p.first, 0);
  EXPECT_EQ(p.second, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(traj.StateAt(i).time, 0);
  }
  for (int i = 4; i < traj.size(); ++i) {
    EXPECT_EQ(traj.StateAt(i).time, i - 3);
  }
}

TEST(Traj, PredictFull) {
  const auto imuq = MakeTestImuq(10);

  TrajCfg cfg;
  Trajectory traj(cfg);
  traj.Allocate(8);
  traj.StateAt(traj.segments()).vel = {1, 0, 0};

  auto p = traj.PredictNew(imuq, {1, 9}, 8);
  EXPECT_EQ(p.first, 0);
  EXPECT_EQ(p.second, 8);
  for (int i = 0; i < traj.size(); ++i) {
    EXPECT_EQ(traj.StateAt(i).pos.x(), i);
  }

  p = traj.PredictFull(imuq);
  EXPECT_EQ(p.first, 0);
  EXPECT_EQ(p.second, 8);
  for (int i = 0; i < traj.size(); ++i) {
    EXPECT_EQ(traj.StateAt(i).pos.x(), i);
  }
}

// TEST(TrajTest, TestIntegrate) {
//  // 1,2,3,4,5
//  ImuBuffer buf(5);
//  for (int i = 0; i < 5; ++i) {
//    ImuData d;
//    d.time = i + 1;
//    buf.push_back(d);
//  }

//  ImuQueue imuq;
//  imuq.buf = buf;

//  Trajectory traj;
//  NavState state;
//  state.time = 1;

//  double dt = (5 - state.time) / traj.segments();
//  EXPECT_EQ(traj.size(), 33);
//  EXPECT_EQ(traj.PredictFull(imuq, state, dt), 4);
//  EXPECT_EQ(traj.states().front().time, state.time);
//  EXPECT_EQ(traj.states().back().time, 5);

//  state.time = 0;
//  dt = (6 - state.time) / traj.segments();
//  EXPECT_EQ(traj.PredictFull(imuq, state, dt), 5);
//  EXPECT_EQ(traj.states().front().time, state.time);
//  EXPECT_EQ(traj.states().back().time, 6);
//}

namespace bm = benchmark;

void BM_TrajInterp(bm::State& state) {
  Trajectory traj;
  traj.Allocate(64);
  std::vector<Sophus::SE3f> tfs(1024);

  const auto gsize = static_cast<int>(state.range(0));
  for (auto _ : state) {
    traj.Interp(tfs, tfs.size(), 0, gsize);
    bm::DoNotOptimize(tfs);
    bm::ClobberMemory();
  }
}
BENCHMARK(BM_TrajInterp)->Arg(0)->Arg(1);

void BM_TrajInterp2(bm::State& state) {
  Trajectory traj;
  traj.Allocate(64);
  std::vector<Sophus::SE3f> tfs(1024);

  const auto gsize = static_cast<int>(state.range(0));
  for (auto _ : state) {
    traj.Interp2(tfs, tfs.size(), 0, gsize);
    bm::DoNotOptimize(tfs);
    bm::ClobberMemory();
  }
}
BENCHMARK(BM_TrajInterp2)->Arg(0)->Arg(1);

}  // namespace
}  // namespace sv::rofl
