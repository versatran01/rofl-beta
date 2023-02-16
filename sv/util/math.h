#pragma once

#include <glog/logging.h>

#include <Eigen/Cholesky>
#include <type_traits>

namespace sv {

static constexpr auto kNaNF = std::numeric_limits<float>::quiet_NaN();
static constexpr auto kNaND = std::numeric_limits<double>::quiet_NaN();
static constexpr auto kPiF = static_cast<float>(M_PI);
static constexpr auto kTauF = static_cast<float>(M_PI * 2);
static constexpr auto kPiD = M_PI;
static constexpr auto kTauD = M_PI * 2;

constexpr bool IsPowerOf2(int n) noexcept {
  const auto log2 = std::log2(n);
  return std::ceil(log2) == std::floor(log2);
}

template <typename T>
constexpr int Round(T d) noexcept {
  return static_cast<int>(std::lround(d));
}

template <typename T>
constexpr int Round2(T d) noexcept {
  return static_cast<int>(d > 0.0 ? d + 0.5 : d - 0.5);
}

template <typename T>
constexpr T Sq(T x) noexcept {
  return x * x;
}

template <typename T>
T Deg2Rad(T deg) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  return deg / 180.0 * M_PI;
}

template <typename T>
T Rad2Deg(T rad) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  return rad / M_PI * 180.0;
}

/// @brief Precomputed sin and cos
template <typename T>
struct SinCos {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  SinCos(T rad = 0) : sin{std::sin(rad)}, cos{std::cos(rad)} {}

  T sin{};
  T cos{};
};

using SinCosF = SinCos<float>;
using SinCosD = SinCos<double>;

/// @struct Running mean and variance
template <typename T, int N>
struct MeanVar {
  using Vector = Eigen::Matrix<T, N, 1>;

  int n{0};
  Vector mean{Vector::Zero()};
  Vector var_sum_{Vector::Zero()};

  /// @brief compute covariance
  Vector Var() const noexcept { return var_sum_ / (n - 1); }

  /// @brief whether result is ok
  bool ok() const noexcept { return n > 1; }

  void Add(const Vector& x) noexcept {
    ++n;
    const Vector dx = x - mean;
    const Vector dx_n = dx / n;
    mean += dx_n;
    var_sum_.noalias() += (n - 1.0) * dx_n.cwiseProduct(dx);
  }

  void Reset() noexcept {
    n = 0;
    mean.setZero();
    var_sum_.setZero();
  }
};

using MeanVar3f = MeanVar<float, 3>;
using MeanVar3d = MeanVar<double, 3>;

/// @struct Running mean and covariance
/// https://stackoverflow.com/questions/37809790/running-one-pass-calculation-of-covariance
template <typename T, int N>
struct MeanCovar {
  using Matrix = Eigen::Matrix<T, N, N>;
  using Vector = Eigen::Matrix<T, N, 1>;
  using MatrixXCRef = Eigen::Ref<const Eigen::Matrix<T, N, Eigen::Dynamic>>;

  int n{0};
  Vector mean{Vector::Zero()};
  Matrix covar_sum{Matrix::Zero()};

  /// @brief compute covariance
  Matrix Covar() const noexcept { return covar_sum / (n - 1); }

  /// @brief whether result is ok
  bool ok() const noexcept { return n > 1; }

  void Add(const Vector& x) noexcept {
    ++n;
    const Vector diff = x - mean;
    const Vector dx_n = diff / n;
    mean += dx_n;
    covar_sum.noalias() += ((n - 1) * dx_n) * diff.transpose();
  }

  void Set(const MatrixXCRef& X) noexcept {
    n = static_cast<int>(X.cols());
    mean = X.rowwise().mean();
    covar_sum.noalias() =
        (X.colwise() - mean) * (X.colwise() - mean).transpose();
  }

  void Reset() noexcept {
    n = 0;
    mean.setZero();
    covar_sum.setZero();
  }
};

using MeanCovar3f = MeanCovar<float, 3>;
using MeanCovar3d = MeanCovar<double, 3>;

/// @brief force the axis to be right handed for 3D
/// @details sometimes eigvecs has det -1 (reflection), this makes it a rotation
/// @ref
/// https://docs.ros.org/en/noetic/api/rviz/html/c++/covariance__visual_8cpp_source.html
void MakeRightHanded(Eigen::Vector3d& eigvals, Eigen::Matrix3d& eigvecs);

/// @brief Computes matrix square root using Cholesky A = LL' = U'U
template <typename T, int N>
Eigen::Matrix<T, N, N> MatrixSqrtUtU(const Eigen::Matrix<T, N, N>& A) {
  return A.template selfadjointView<Eigen::Upper>().llt().matrixU();
}

/// A (real) closed interval, boost interval is too heavy
template <typename T>
struct Interval {
  Interval() = default;
  Interval(const T& left, const T& right) noexcept
      : left_(left), right_(right) {
    CHECK_LE(left, right);
  }

  T left_, right_;

  const T& a() const noexcept { return left_; }
  const T& b() const noexcept { return right_; }
  T width() const noexcept { return b() - a(); }
  bool empty() const noexcept { return b() <= a(); }
  bool ContainsClosed(const T& v) const noexcept {
    return (a() <= v) && (v <= b());
  }
  bool ContainsOpen(const T& v) const noexcept {
    return (a() < v) && (v < b());
  }

  /// Whether this interval contains other
  bool ContainsClosed(const Interval<T>& other) const noexcept {
    return a() <= other.a() && other.b() <= b();
  }

  /// Normalize v to [0, 1], assumes v in [left, right]
  /// Only enable if we have floating type
  T Normalize(const T& v) const noexcept {
    static_assert(std::is_floating_point_v<T>, "Must be floating point");
    return (v - a()) / width();
  }

  /// InvNormalize v to [left, right], assumes v in [0, 1]
  T InvNormalize(const T& v) const noexcept {
    static_assert(std::is_floating_point_v<T>, "Must be floating point");
    return v * width() + a();
  }
};

using IntervalF = Interval<float>;
using IntervalD = Interval<double>;

/// @brief Polynomial approximation to asin
template <typename T>
T AsinApprox(T x) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  const auto x2 = x * x;
  return x * (1 + x2 * (1 / 6.0 + x2 * (3.0 / 40.0 + x2 * 5.0 / 112.0)));
}

template <typename T>
constexpr T AsinApprox3rd(T x) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  constexpr T pi_2 = M_PI_2;
  constexpr T a0 = 1.5707288;
  constexpr T a1 = -0.2121144;
  constexpr T a2 = 0.0742610;
  constexpr T a3 = -0.0187293;

  const auto neg = x < 0;
  x = std::abs(x);
  const auto x2 = x * x;
  auto res = pi_2 - std::sqrt(1 - x) * (a0 + a1 * x + a2 * x2 + a3 * x2 * x);
  if (neg) res = -res;
  return res;
}

template <typename T>
T Atan2Approx(T y, T x) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/atan2.html
  // Volkan SALMA
  constexpr T kPi3_4 = M_PI_4 * 3;
  constexpr T kPi_4 = M_PI_4;

  T r;
  T angle;
  const T abs_y = std::abs(y) + 1e-10;  // kludge to prevent 0/0 condition

  if (x < T(0)) {
    r = (x + abs_y) / (abs_y - x);
    angle = kPi3_4;
  } else {
    r = (x - abs_y) / (x + abs_y);
    angle = kPi_4;
  }
  angle += (0.1963 * r * r - 0.9817) * r;
  return y < T(0) ? -angle : angle;
}

/// @brief Polynomial approximation to atan
/// @details https://mazzo.li/posts/vectorized-atan2.html
template <typename T>
constexpr T AtanApprox6th(T x) noexcept {
  static_assert(std::is_floating_point_v<T>, "");
  constexpr T a1 = 0.99997726;
  constexpr T a3 = -0.33262347;
  constexpr T a5 = 0.19354346;
  constexpr T a7 = -0.11643287;
  constexpr T a9 = 0.05265332;
  constexpr T a11 = -0.01172120;

  const auto x2 = x * x;
  return x * (a1 + x2 * (a3 + x2 * (a5 + x2 * (a7 + x2 * (a9 + x2 * a11)))));
}

/// @brief Atan2 based on atan approx
template <typename T>
constexpr T Atan2Approx6th(T y, T x) noexcept {
  constexpr T pi = M_PI;
  constexpr T pi_2 = M_PI_2;
  const auto swap = std::abs(x) < std::abs(y);
  const auto ratio = swap ? (x / y) : (y / x);

  // Approximate atan
  auto res = AtanApprox6th(ratio);

  // If swapped, adjust atan output
  if (swap) res = ratio >= T(0) ? pi_2 - res : -pi_2 - res;
  if (x < T(0)) res += y >= T(0) ? pi : -pi;
  return res;
}

}  // namespace sv
