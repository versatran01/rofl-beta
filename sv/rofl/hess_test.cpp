#include "sv/rofl/hess.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

const Eigen::Matrix3d I33 = Eigen::Matrix3d::Identity();

TEST(Hess, Hess1Ctor) {
  Hess1 hess;
  EXPECT_EQ(hess.c, 0);
  EXPECT_EQ(hess.n, 0);
}

TEST(Hess, Hess1OpPlus) {
  Hess1 h1;
  h1.c = 1;
  h1.n = 1;
  Hess1 h2;
  h2.c = 2;
  h2.n = 2;

  const auto h3 = h1 + h2;
  EXPECT_EQ(h3.c, 3);
  EXPECT_EQ(h3.n, 3);
}

TEST(Hess, Hess1Solve) {
  Hess1 hess;
  hess.H.setIdentity();
  hess.b.setOnes();
  hess.n = 10;
  EXPECT_EQ(hess.Solve(), Hess1::Vector6d::Ones());
}

namespace bm = benchmark;

void BM_Hess1Add(bm::State& state) {
  Hess1 hess;
  const auto J = Hess1::Matrix36d::Ones().eval();
  const auto W = Eigen::Matrix3d::Identity().eval();
  const auto r = Eigen::Vector3d::Zero().eval();

  for (auto _ : state) {
    hess.Add(J, W, r);
    bm::DoNotOptimize(hess);
  }
}
BENCHMARK(BM_Hess1Add);

}  // namespace
}  // namespace sv::rofl
