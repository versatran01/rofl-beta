#include "sv/rofl/proj.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

TEST(Proj, Ctor) {
  Projection p{{1024, 64}};

  EXPECT_EQ(p.rows(), 64);
  EXPECT_EQ(p.cols(), 1024);
  EXPECT_EQ(p.elevs.size(), 64);
  EXPECT_EQ(p.azims.size(), 1024);

  std::cout << p.Repr() << "\n";
}

TEST(Proj, ToRowAndToCol) {
  Projection p{{8, 8}, Deg2Rad(70.0)};
  EXPECT_EQ(p.elev_max, Deg2Rad(35.0));
  EXPECT_EQ(p.elev_delta, Deg2Rad(10.0));
  EXPECT_DOUBLE_EQ(p.elevs.front().sin, std::sin(Deg2Rad(35.0)));
  EXPECT_DOUBLE_EQ(p.elevs.front().cos, std::cos(Deg2Rad(35.0)));
  EXPECT_DOUBLE_EQ(p.elevs.back().sin, std::sin(Deg2Rad(-35.0)));
  EXPECT_DOUBLE_EQ(p.elevs.back().cos, std::cos(Deg2Rad(-35.0)));

  EXPECT_DOUBLE_EQ(p.azim_delta, Deg2Rad(45.0));
  EXPECT_DOUBLE_EQ(p.azims.front().sin, std::sin(Deg2Rad(0.0)));
  EXPECT_DOUBLE_EQ(p.azims.front().cos, std::cos(Deg2Rad(0.0)));
  EXPECT_DOUBLE_EQ(p.azims.back().sin, std::sin(Deg2Rad(45.0)));
  EXPECT_DOUBLE_EQ(p.azims.back().cos, std::cos(Deg2Rad(45.0)));

  //     0       1       2       3       4       5       6       7
  // |---*---|---*---|---*---|---*---|---*---|---*---|---*---|---*---|
  // 40     30      20      10       0      -10     -20     -30     -40

  auto deg2row = [&](double deg) {
    return p.ToRowI(std::sin(Deg2Rad(deg)), 1);
  };

  EXPECT_EQ(deg2row(45.00), -1);
  EXPECT_EQ(deg2row(40.01), -1);

  EXPECT_EQ(deg2row(39.99), 0);
  EXPECT_EQ(deg2row(35.00), 0);
  EXPECT_EQ(deg2row(30.01), 0);

  EXPECT_EQ(deg2row(29.99), 1);
  EXPECT_EQ(deg2row(25.00), 1);
  EXPECT_EQ(deg2row(20.01), 1);

  EXPECT_EQ(deg2row(0.01), 3);
  EXPECT_EQ(deg2row(-0.01), 4);

  EXPECT_EQ(deg2row(-30.01), 7);
  EXPECT_EQ(deg2row(-35.00), 7);
  EXPECT_EQ(deg2row(-39.99), 7);

  EXPECT_EQ(deg2row(-40.01), 8);
  EXPECT_EQ(deg2row(-45.00), 8);

  auto deg2col = [&](double deg) {
    const auto rad = Deg2Rad(deg);
    return p.ToColI(std::cos(rad), std::sin(rad));
  };

  EXPECT_EQ(deg2col(0), 8);
  EXPECT_EQ(deg2col(0.01), 8);

  EXPECT_EQ(deg2col(44.99), 7);
  EXPECT_EQ(deg2col(45), 7);
  EXPECT_EQ(deg2col(45.01), 7);

  EXPECT_EQ(deg2col(180), 4);
  EXPECT_EQ(deg2col(315), 1);

  EXPECT_EQ(deg2col(359), 0);
}

namespace bm = benchmark;

void BM_ProjToRowD(bm::State& state) {
  Projection p{{1024, 64}};

  const auto x = static_cast<double>(state.range(0));
  for (auto _ : state) {
    bm::DoNotOptimize(p.ToRowD(x, x));
  }
}
BENCHMARK(BM_ProjToRowD)->Arg(1);

void BM_ProjToRowI(bm::State& state) {
  Projection p{{1024, 64}};

  const auto x = static_cast<double>(state.range(0));
  for (auto _ : state) {
    bm::DoNotOptimize(p.ToRowI(x, x));
  }
}
BENCHMARK(BM_ProjToRowI)->Arg(1);

void BM_ProjToColD(bm::State& state) {
  Projection p{{1024, 64}};

  const auto x = static_cast<double>(state.range(0));
  for (auto _ : state) {
    bm::DoNotOptimize(p.ToColD(x, x));
  }
}
BENCHMARK(BM_ProjToColD)->Arg(1);

void BM_ProjToColI(bm::State& state) {
  Projection p{{1024, 64}};

  const auto x = static_cast<double>(state.range(0));
  for (auto _ : state) {
    bm::DoNotOptimize(p.ToColI(x, x));
  }
}
BENCHMARK(BM_ProjToColI)->Arg(1);

}  // namespace
}  // namespace sv::rofl
