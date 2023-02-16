#include "sv/rofl/gicp.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

const Eigen::Matrix3f I33 = Eigen::Matrix3f::Identity();

TEST(Gicp, RigidBadPano) {
  // A bad pano should not have any match

  auto sweep = MakeTestSweep({1024, 64});
  sweep.CalcRangeGrad2();

  // Select from grid
  SweepGrid grid;
  grid.Allocate(sweep.size2d());
  const auto n_sel = grid.Select(sweep);
  ASSERT_EQ(n_sel, grid.size() / 2);

  // Make a test pano (currently bad)
  const Projection lidar({1024, 256});
  auto pano = MakeTestPano(lidar.size2d(), 0, 0);

  // For a bad pano we will have a bad hessian
  GicpSolver solver;
  solver.Allocate(grid.size2d());

  //  Sophus::SE3d dtf;
  //  const auto hess = solver.BuildRigid(grid, pano, lidar, dtf, 0);
  //  EXPECT_EQ(hess.n, 0);
  //  EXPECT_EQ(hess.c, 0);
}

namespace bm = benchmark;

void BM_MatchCalcInfo(bm::State& state) {
  GicpMatch match;
  for (int i = 0; i < 10; ++i) {
    match.mc.Add(Eigen::Vector3f::Random());
  }

  for (auto _ : state) {
    match.CalcInfo(I33, I33);
    bm::DoNotOptimize(match);
  }
}
BENCHMARK(BM_MatchCalcInfo);

}  // namespace
}  // namespace sv::rofl
