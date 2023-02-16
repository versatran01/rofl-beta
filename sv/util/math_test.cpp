#include "sv/util/math.h"

#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

Eigen::Matrix3d CalCovar3d(const Eigen::Matrix3Xd& X) {
  const Eigen::Vector3d m = X.rowwise().mean();   // mean
  const Eigen::Matrix3Xd Xm = (X.colwise() - m);  // centered
  return ((Xm * Xm.transpose()) / (X.cols() - 1));
}

Eigen::Vector3d CalVar3d(const Eigen::Matrix3Xd& X) {
  const Eigen::Vector3d m = X.rowwise().mean();   // mean
  const Eigen::Matrix3Xd Xm = (X.colwise() - m);  // centered
  return Xm.cwiseProduct(Xm).rowwise().sum() / (X.cols() - 1);
}

TEST(Math, AngleConversion) {
  EXPECT_DOUBLE_EQ(Deg2Rad(0.0), 0.0);
  EXPECT_DOUBLE_EQ(Deg2Rad(90.0), M_PI / 2.0);
  EXPECT_DOUBLE_EQ(Deg2Rad(180.0), M_PI);
  EXPECT_DOUBLE_EQ(Deg2Rad(360.0), M_PI * 2);
  EXPECT_DOUBLE_EQ(Deg2Rad(-180.0), -M_PI);

  EXPECT_DOUBLE_EQ(Rad2Deg(0.0), 0.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI / 2), 90.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI), 180.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI * 2), 360.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(-M_PI), -180.0);
}

TEST(Math, MatrixSqrtUtU) {
  Eigen::Matrix3Xf X = Eigen::Matrix3Xf::Random(3, 100);
  const Eigen::Matrix3f A = X * X.transpose();
  const Eigen::Matrix3f U = MatrixSqrtUtU(A);
  const Eigen::Matrix3f UtU = U.transpose() * U;
  EXPECT_TRUE(A.isApprox(UtU));
}

TEST(Math, MeanVar) {
  for (int i = 3; i < 50; i += 10) {
    const auto X = Eigen::Matrix3Xd::Random(3, i).eval();
    const auto var0 = CalVar3d(X);
    const auto m = X.rowwise().mean().eval();

    MeanVar3d mv;
    for (int j = 0; j < X.cols(); ++j) mv.Add(X.col(j));
    const auto var1 = mv.Var();

    EXPECT_TRUE(var0.isApprox(var1));
    EXPECT_TRUE(mv.mean.isApprox(m));
  }
}

TEST(Math, MeanCovar) {
  for (int i = 3; i < 50; i += 10) {
    const auto X = Eigen::Matrix3Xd::Random(3, i).eval();
    const auto cov = CalCovar3d(X);
    const auto m = X.rowwise().mean().eval();

    MeanCovar3d mc1;
    for (int j = 0; j < X.cols(); ++j) mc1.Add(X.col(j));
    const auto cov1 = mc1.Covar();

    MeanCovar3d mc2;
    mc2.Set(X);
    const auto cov2 = mc2.Covar();

    EXPECT_TRUE(cov1.isApprox(cov));
    EXPECT_TRUE(cov2.isApprox(cov));
    EXPECT_TRUE(mc1.mean.isApprox(m));
    EXPECT_TRUE(mc2.mean.isApprox(m));
  }
}

TEST(Math, AsinApprox) {
  const int n = 1000;
  const auto bound = Deg2Rad(85.0);
  const auto delta = bound * 2 / n;

  for (int i = 0; i < n; ++i) {
    const auto theta = -bound + delta * i;
    const auto sin = std::sin(theta);
    const auto asin0 = std::asin(sin);
    const auto asin1 = AsinApprox3rd(sin);
    const auto diff = std::abs(asin0 - asin1);
    EXPECT_LE(diff, 7e-5) << fmt::format(
        "theta: {},  asin0: {}, asin1: {}", theta, asin0, asin1);
  }
}

TEST(Math, Atan2Approx) {
  const int n = 1000;
  const auto delta = kTauD / n;

  for (int i = 0; i < n; ++i) {
    const auto theta = delta * i;
    const auto x = std::cos(theta);
    const auto y = std::sin(theta);
    const auto atan20 = std::atan2(y, x);
    const auto atan21 = Atan2Approx6th(y, x);
    const auto diff = std::abs(atan20 - atan21);
    EXPECT_LE(diff, 2e-6) << fmt::format(
        "theta: {}, atan2_0: {}, atan2_1: {}", theta, atan20, atan21);
  }
}

TEST(Math, Round) {
  const int n = 500;
  const double bound = 2;
  const auto delta = bound * 2 / n;

  for (int i = 0; i < n; ++i) {
    const auto d = -bound + delta * i;
    const auto i0 = Round(d);
    const auto i1 = Round2(d);
    EXPECT_EQ(i0, i1) << fmt::format("d: {}, i0: {}, i1: {}", d, i0, i1);
  }
}

/// ============================================================================
namespace bm = benchmark;

void BM_Covariance(bm::State& state) {
  const auto X = Eigen::Matrix3Xd::Random(3, state.range(0)).eval();

  for (auto _ : state) {
    bm::DoNotOptimize(CalCovar3d(X));
  }
}
BENCHMARK(BM_Covariance)->Range(8, 32);

void BM_MeanCovar3fAdd(bm::State& state) {
  const auto X = Eigen::Matrix3Xf::Random(3, state.range(0)).eval();

  MeanCovar3f mc;
  for (auto _ : state) {
    for (int i = 0; i < X.cols(); ++i) mc.Add(X.col(i));
    bm::DoNotOptimize(mc.Covar());
  }
}
BENCHMARK(BM_MeanCovar3fAdd)->Range(8, 32);

void BM_MeanCovar3dAdd(bm::State& state) {
  const auto X = Eigen::Matrix3Xd::Random(3, state.range(0)).eval();

  MeanCovar3d mc;
  for (auto _ : state) {
    for (int i = 0; i < X.cols(); ++i) mc.Add(X.col(i));
    const auto cov = mc.Covar();
    bm::DoNotOptimize(cov);
  }
}
BENCHMARK(BM_MeanCovar3dAdd)->Range(8, 32);

void BM_MeanCovar3dSet(bm::State& state) {
  const auto X = Eigen::Matrix3Xd::Random(3, state.range(0)).eval();

  MeanCovar3d mc;
  for (auto _ : state) {
    mc.Set(X);
    const auto cov = mc.Covar();
    bm::DoNotOptimize(cov);
  }
}
BENCHMARK(BM_MeanCovar3dSet)->Range(8, 32);

std::vector<double> MakeAsinData(int n) {
  std::vector<double> data(n);
  const auto delta = 2.0 / n;
  for (int i = 0; i < n; ++i) {
    const auto theta = -1 + i * delta;
    data[i] = std::sin(theta);
  }
  return data;
}

void BM_AsinGlibc(bm::State& state) {
  const auto data = MakeAsinData(100);

  for (auto _ : state) {
    for (const auto& d : data) {
      bm::DoNotOptimize(std::asin(d));
    }
  }
}
BENCHMARK(BM_AsinGlibc);

void BM_AsinApprox(bm::State& state) {
  const auto data = MakeAsinData(100);

  for (auto _ : state) {
    for (const auto& d : data) {
      bm::DoNotOptimize(AsinApprox(d));
    }
  }
}
BENCHMARK(BM_AsinApprox);

void BM_AsinApprox3rd(bm::State& state) {
  const auto data = MakeAsinData(100);

  for (auto _ : state) {
    for (const auto& d : data) {
      bm::DoNotOptimize(AsinApprox3rd(d));
    }
  }
}
BENCHMARK(BM_AsinApprox3rd);

std::vector<Eigen::Vector2d> MakeAtan2Data(int n) {
  std::vector<Eigen::Vector2d> data(n);
  const auto delta = M_PI * 2 / n;
  for (int i = 0; i < n; ++i) {
    const auto theta = delta * i;
    data[i].x() = std::cos(theta);
    data[i].y() = std::sin(theta);
  }
  return data;
}

void BM_Atan2Glibc(bm::State& state) {
  const auto data = MakeAtan2Data(100);

  for (auto _ : state) {
    for (const auto& d : data) {
      bm::DoNotOptimize(std::atan2(d.y(), d.x()));
    }
  }
}
BENCHMARK(BM_Atan2Glibc);

void BM_Atan2Approx(bm::State& state) {
  const auto data = MakeAtan2Data(100);

  for (auto _ : state) {
    for (const auto& d : data) {
      bm::DoNotOptimize(Atan2Approx(d.y(), d.x()));
    }
  }
}
BENCHMARK(BM_Atan2Approx);

void BM_Atan2Approx6th(bm::State& state) {
  const auto data = MakeAtan2Data(100);

  for (auto _ : state) {
    for (const auto& d : data) {
      bm::DoNotOptimize(Atan2Approx6th(d.y(), d.x()));
    }
  }
}
BENCHMARK(BM_Atan2Approx6th);

}  // namespace
}  // namespace sv
