#include "sv/util/eigen.h"

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>

namespace sv {
namespace {

namespace bm = benchmark;
using Vector3d = Eigen::Vector3d;
using Vector10d = VectorNd<10>;
using Matrix10d = MatrixMNd<10, 10>;

TEST(Eigen, Hat3) {
  Vector3d x(1, 2, 3);

  const auto S = Hat3d(x);
  EXPECT_EQ(S.transpose(), -S);
  const auto S1 = Hat3d(-x);
  EXPECT_EQ(S1, -S);
}

/// @brief Invert PSD matrix, taken from ceres
template <int N>
void InvertPSDMatrix(const MatrixMNd<N, N>& m, MatrixXdRef m_inv) {
  const auto size = m.rows();

  // If the matrix can be assumed to be full rank, then if it is small
  // (< 5) and fixed size, use Eigen's optimized inverse()
  // implementation.
  //
  // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html#title3
  if constexpr (0 < N && N < 5) {
    m_inv = m.inverse();
    return;
  }
  m_inv = m.template selfadjointView<Eigen::Upper>().llt().solve(
      MatrixMNd<N, N>::Identity(size, size));
}

TEST(Eigen, SafeCwiseInv) {
  VectorXd x0 = VectorXd::Zero(4);
  x0(0) = 1;
  x0(2) = 1;
  VectorXd x1 = x0;

  SafeCwiseInverse(x1);
  EXPECT_EQ(x1, x0);
}

TEST(Eigen, StableRotateBlockUpperLeftSize1) {
  Eigen::Matrix3d H;
  // clang-format off
  H << 1, 2, 3,
       4, 5, 6,
       7, 8, 9;
  // clang-format on
  Eigen::Vector3d b;
  b << 1, 2, 3;

  StableRotateBlockTopLeft(H, b, /*ind*/ 2, /*size*/ 1);

  Eigen::Matrix3d H1;
  // clang-format off
  H1 << 9, 7, 8,
        3, 1, 2,
        6, 4, 5;
  // clang-format on
  EXPECT_EQ(H, H1);
  EXPECT_EQ(b, Eigen::Vector3d(3, 1, 2));
}

TEST(Eigen, StableRotateBlockUpperLeftSize2) {
  // 0, 1, 2,
  // 3, 4, 5,
  // 6, 7, 8

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  int k = 0;
  for (int i = 0; i < 3; ++i) {
    b.segment<2>(i * 2).setConstant(i);
    for (int j = 0; j < 3; ++j) {
      H.block<2, 2>(i * 2, j * 2).setConstant(k);
      ++k;
    }
  }

  LOG(INFO) << "\n" << H;
  LOG(INFO) << "\n" << b.transpose();

  StableRotateBlockTopLeft(H, b, 2, 2);

  LOG(INFO) << "\n" << H;
  LOG(INFO) << "\n" << b.transpose();

  const auto H00 = H.topLeftCorner<2, 2>().eval();
  EXPECT_EQ(H00, Eigen::Matrix2d::Constant(8));

  const auto b0 = b.head<2>().eval();
  EXPECT_EQ(b0, Eigen::Vector2d::Constant(2));
}

TEST(Eigen, StableRotateBlockUpperLeft) {
  using Matrix6d = MatrixMNd<6, 6>;
  using Vector6d = MatrixMNd<6, 1>;
  MatrixXd A = MatrixXd::Random(6, 100);
  Matrix6d H = A * A.transpose();

  Vector6d x;
  x << 1, 2, 3, 4, 5, 6;

  Vector6d b = H * x;
  Vector6d x0 = H.selfadjointView<Eigen::Lower>().llt().solve(b);
  EXPECT_EQ(x.isApprox(x0), true);

  // Now we rotate
  StableRotateBlockTopLeft(H, b, 1, 2);
  Vector6d x1 = H.selfadjointView<Eigen::Lower>().llt().solve(b);
  LOG(INFO) << x1.transpose();
}

TEST(Eigen, FillLowerTriangular) {
  Eigen::Matrix3d M;
  // clang-format off
  M << 1, 2, 3,
       4, 5, 6,
       7, 8, 9;
  // clang-format on
  M = M.triangularView<Eigen::Upper>();

  Eigen::Matrix3d Mu;
  // clang-format off
  Mu << 1, 2, 3,
        0, 5, 6,
        0, 0, 9;
  // clang-format on
  EXPECT_EQ(M, Mu);

  Eigen::Matrix3d M0;
  // clang-format off
  M0 << 1, 2, 3,
        2, 5, 6,
        3, 6, 9;
  // clang-format on

  FillLowerTriangular(M);
  EXPECT_EQ(M, M0);
}

TEST(Eigen, FillUpperTriangular) {
  Eigen::Matrix3d M;
  // clang-format off
  M << 1, 2, 3,
       4, 5, 6,
       7, 8, 9;
  // clang-format on
  M = M.triangularView<Eigen::Lower>();

  Eigen::Matrix3d Ml;
  // clang-format off
  Ml << 1, 0, 0,
        4, 5, 0,
        7, 8, 9;
  // clang-format on
  EXPECT_EQ(M, Ml);

  Eigen::Matrix3d M0;
  // clang-format off
  M0 << 1, 4, 7,
        4, 5, 8,
        7, 8, 9;
  // clang-format on

  FillUpperTriangular(M);
  EXPECT_EQ(M, M0);
}

TEST(Eigen, MakeSymmetric) {
  MatrixXd A = MatrixXd::Random(5, 5);
  MakeSymmetric(A);
  CHECK_EQ(A, A.transpose()) << "\n" << A;
}

TEST(Eigen, MargTopLeftBlock) {
  MatrixXd Hsc = MatrixXd::Zero(4, 4);
  Hsc.topLeftCorner<2, 2>().setIdentity();
  Hsc.topRightCorner<2, 2>().setOnes();
  Hsc.bottomLeftCorner<2, 2>().setOnes();
  Hsc.bottomRightCorner<2, 2>().setIdentity();

  VectorXd bsc = VectorXd::Ones(4);
  MatrixXd Hpr = MatrixXd::Zero(2, 2);
  VectorXd bpr = VectorXd::Zero(2);

  MargTopLeftBlock(Hsc, bsc, Hpr, bpr, 2);

  MatrixXd Hpr0(2, 2);
  Hpr0 << -1, -2, -2, -1;
  VectorXd bpr0(2);
  bpr0 << -1, -1;
  EXPECT_EQ(Hpr, Hpr0);
  EXPECT_EQ(bpr, bpr0);
}

TEST(Eigen, MargTopLeftBlock2) {
  MatrixXd A = MatrixXd::Random(10, 40);
  MatrixXd Hsc = A * A.transpose();
  MakeSymmetric(Hsc);
  VectorXd bsc = VectorXd::Ones(10);

  MatrixXd Hpr(5, 5);
  VectorXd bpr(5);

  MargTopLeftBlock(Hsc, bsc, Hpr, bpr, 5);
  EXPECT_EQ(Hpr, Hpr.transpose()) << "\n" << Hpr;
  EXPECT_EQ(Hpr.isApprox(Hpr.transpose()), true) << "\n" << Hpr;
}

/// ============================================================================
void BM_CwiseInverse(bm::State& state) {
  Eigen::VectorXd v(state.range(0));
  v.setOnes();

  for (auto _ : state) {
    v = v.cwiseInverse();
    bm::DoNotOptimize(v);
  }
}
BENCHMARK(BM_CwiseInverse)->Arg(64)->Arg(1024)->Arg(2048)->Arg(4096);

void BM_SafeCwiseInverse(bm::State& state) {
  Eigen::VectorXd v(state.range(0));
  v.setZero();
  for (int i = 0; i < v.size(); i += 2) {
    v[i] = 1;
  }

  for (auto _ : state) {
    SafeCwiseInverse(v);
    bm::DoNotOptimize(v);
  }
}
BENCHMARK(BM_SafeCwiseInverse)->Arg(64)->Arg(1024)->Arg(2048)->Arg(4096);

/// ============================================================================
void BM_InvertPSDEigen(bm::State& state) {
  const auto n = state.range(0);
  Eigen::MatrixXd m(n, n);
  Eigen::MatrixXd m_inv(n, n);
  m.setIdentity();

  for (auto _ : state) {
    m_inv = m.inverse();
    bm::DoNotOptimize(m_inv);
  }
}
BENCHMARK(BM_InvertPSDEigen)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_InvertPSDCeres(bm::State& state) {
  const auto n = state.range(0);
  Eigen::MatrixXd m(n, n);
  Eigen::MatrixXd m_inv(n, n);
  m.setIdentity();

  for (auto _ : state) {
    InvertPSDMatrix(m, m_inv);
    bm::DoNotOptimize(m_inv);
  }
}
BENCHMARK(BM_InvertPSDCeres)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

void BM_FillLowerTriangular(bm::State& state) {
  const auto size = state.range(0);
  MatrixXd U = MatrixXd::Random(size, size);

  for (auto _ : state) {
    FillLowerTriangular(U);
    bm::DoNotOptimize(U);
  }
}
BENCHMARK(BM_FillLowerTriangular)->Arg(20)->Arg(40)->Arg(60);

void BM_AssignSelfAdjoint(bm::State& state) {
  const auto size = state.range(0);
  MatrixXd U = MatrixXd::Random(size, size);
  MatrixXd X(size, size);

  for (auto _ : state) {
    X = U.selfadjointView<Eigen::Upper>();
    bm::DoNotOptimize(X);
  }
}
BENCHMARK(BM_AssignSelfAdjoint)->Arg(20)->Arg(40)->Arg(60);

void BM_SolveDimLLT(bm::State& state) {
  MatrixXd X = MatrixXd::Random(10, 100);
  Matrix10d H = X * X.transpose();
  Vector10d x = Vector10d::Random();
  Vector10d b = H * x;
  auto dim = state.range(0);

  for (auto _ : state) {
    Vector10d xs;
    xs.setZero();
    xs.head(dim) =
        H.topLeftCorner(dim, dim).selfadjointView<Eigen::Lower>().llt().solve(
            b.head(dim));
    bm::DoNotOptimize(xs);
  }
}
BENCHMARK(BM_SolveDimLLT)->Arg(6)->Arg(8)->Arg(10);

void BM_SolveLDLT(bm::State& state) {
  MatrixXd X = MatrixXd::Random(10, 100);
  Matrix10d H = X * X.transpose();
  Vector10d x = Vector10d::Random();
  Vector10d b = H * x;

  for (auto _ : state) {
    const auto xs = H.selfadjointView<Eigen::Lower>().ldlt().solve(b);
    bm::DoNotOptimize(xs);
  }
}
BENCHMARK(BM_SolveLDLT);

// RowMajor X * X' is actually slightly slower
void BM_MatMulColMajor(bm::State& state) {
  MatrixXd X(50, 4000);
  X.setOnes();
  MatrixXd Y(50, 50);

  for (auto _ : state) {
    Y.noalias() = X * X.transpose();
    bm::DoNotOptimize(Y);
  }
}
BENCHMARK(BM_MatMulColMajor);

void BM_MatMulRowMajor(bm::State& state) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(
      50, 4000);
  X.setOnes();
  MatrixXd Y(50, 50);

  for (auto _ : state) {
    Y.noalias() = X * X.transpose();
    bm::DoNotOptimize(Y);
  }
}
BENCHMARK(BM_MatMulRowMajor);

}  // namespace
}  // namespace sv
