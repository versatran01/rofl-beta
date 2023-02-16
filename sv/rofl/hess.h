#include <sophus/se3.hpp>

namespace sv::rofl {

/// @brief Hessian of a single delta pose for rigid icp
struct Hess1 {
  using Vector6d = Eigen::Matrix<double, 6, 1>;  // [rot, trans]
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Matrix36d = Eigen::Matrix<double, 3, 6>;  // J
  using Matrix63d = Eigen::Matrix<double, 6, 3>;  // Jt

  int n{};     // number of costs
  double c{};  // total costs (sum r^2)

  Matrix6d H{Matrix6d::Zero()};
  Vector6d b{Vector6d::Zero()};

  std::string Repr() const;

  Hess1& operator+=(const Hess1& rhs);
  friend Hess1 operator+(Hess1 lhs, const Hess1& rhs) { return lhs += rhs; }

  /// @brief Add a single cost from match, Jt version is faster
  void Add(const Matrix36d& J,
           const Eigen::Matrix3d& W,
           const Eigen::Vector3d& r) {
    const Matrix63d JtW = J.transpose() * W;
    H.noalias() += JtW * J;
    b.noalias() -= JtW * r;

    c += r.squaredNorm();
    ++n;
  }

  /// @brief Solve
  Vector6d Solve() const;
};

}  // namespace sv::rofl
