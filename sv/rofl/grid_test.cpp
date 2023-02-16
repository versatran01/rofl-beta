#include "sv/rofl/grid.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
constexpr auto inf = std::numeric_limits<float>::quiet_NaN();

TEST(Grid, FindLongestArea) {
  EXPECT_EQ(FindLongestRange(VectorXf{{}}, 1), cv::Range(0, 0));
  EXPECT_EQ(FindLongestRange(VectorXf{{0}}, 1), cv::Range(0, 1));
  EXPECT_EQ(FindLongestRange(VectorXf{{0, 0}}, 1), cv::Range(0, 2));
  EXPECT_EQ(FindLongestRange(VectorXf{{2, 2}}, 1), cv::Range(0, 0));
  EXPECT_EQ(FindLongestRange(VectorXf{{0, 2, 0}}, 1), cv::Range(0, 1));
  EXPECT_EQ(FindLongestRange(VectorXf{{2, 0, 0}}, 1), cv::Range(1, 3));
  EXPECT_EQ(FindLongestRange(VectorXf{{0, 2, 0, 0}}, 1), cv::Range(2, 4));
  EXPECT_EQ(FindLongestRange(VectorXf{{2, 0, 0, 0, 2}}, 1), cv::Range(1, 4));
  EXPECT_EQ(FindLongestRange(VectorXf{{2, 0, 2, 0, 0, 0}}, 1), cv::Range(3, 6));
  EXPECT_EQ(FindLongestRange(VectorXf{{2, 0, 0, 0, nan}}, 1), cv::Range(1, 4));
  EXPECT_EQ(FindLongestRange(VectorXf{{inf, 0, 0, 0, nan}}, 1),
            cv::Range(1, 4));
}

TEST(Grid, Select) {
  auto sweep = MakeTestSweep({1024, 64});
  sweep.CalcRangeGrad2();

  SweepGrid grid;
  grid.Allocate(sweep.size2d());

  EXPECT_EQ(grid.Select(sweep), grid.size() / 2);
}

namespace bm = benchmark;

void BM_GridSelect(bm::State& state) {
  auto sweep = MakeTestSweep({1024, 64});
  sweep.CalcRangeGrad2();

  SweepGrid grid;
  grid.Allocate(sweep.size2d());

  const int gsize = static_cast<int>(state.range(0));
  for (auto _ : state) {
    const auto n = grid.Select(sweep, gsize);
    bm::DoNotOptimize(n);
    bm::ClobberMemory();
  }
}
BENCHMARK(BM_GridSelect)->Arg(0)->Arg(1);

}  // namespace
}  // namespace sv::rofl
