#include "sv/rofl/hess.h"

#include <Eigen/Cholesky>

#include "sv/util/logging.h"

namespace sv::rofl {

using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

std::string Hess1::Repr() const {
  return fmt::format("Hessian1(n={}, c={:.4f})", n, c);
}

Hess1& Hess1::operator+=(const Hess1& rhs) {
  if (rhs.n > 0) {
    H += rhs.H;
    b += rhs.b;
    n += rhs.n;
    c += rhs.c;
  }
  return *this;
}

auto Hess1::Solve() const -> Vector6d {
  CHECK_GE(n, 6);
  return H.selfadjointView<Eigen::Lower>().llt().solve(b);
}

}  // namespace sv::rofl
