#pragma once

#include <opencv2/core/types.hpp>

#include "sv/util/math.h"  // Round, SinCos

namespace sv::rofl {

/// @brief Whether this pixel is bad
inline bool IsPixBad(cv::Point px) noexcept { return px.x < 0 || px.y < 0; }

/// @brief This should really be called PanoModel
struct Projection {
  cv::Size size2d_{};
  double elev_max{};
  double elev_delta{};
  double azim_delta{};
  std::vector<SinCosD> elevs{};
  std::vector<SinCosD> azims{};

  Projection() = default;

  /// @brief Create a pano projection model with centered vertical fov and 360
  /// horizontal fov.
  explicit Projection(cv::Size size2d, double vfov = 0.0);

  /// @brief Repr / <<
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const Projection& rhs) {
    return os << rhs.Repr();
  }

  bool empty() const noexcept { return elevs.empty() || azims.empty(); }

  cv::Size size2d() const noexcept { return size2d_; }
  int rows() const noexcept { return size2d_.height; }
  int cols() const noexcept { return size2d_.width; }

  double ToRowD(double z, double r) const {
    //  return (elev_max - std::asin(z / r)) / elev_delta;
    return (elev_max - AsinApprox3rd(z / r)) / elev_delta;
  }

  /// @brief The coordinate of the projection is
  ///  ^ y
  ///  |
  ///  |
  ///  |       | theta (azimuth)
  ///  o -------> x
  ///          | col index
  ///
  /// However, since we want to visualize the pano as an image, col index starts
  /// from +x at 0 and increases clockwise. But theta starts from +x at 0 and
  /// increase counter-clockwise. This is taken care of in construction.
  double ToColD(double x, double y) const {
    // return (std::atan2(y, -x) + kPiD) / azim_delta;
    return (Atan2Approx6th(y, -x) + kPiD) / azim_delta;
  }

  int ToRowI(double z, double r) const { return Round2(ToRowD(z, r)); }
  int ToColI(double x, double y) const { return Round2(ToColD(x, y)); }

  /// @brief Backward from pixel and range to 3d point
  cv::Point3d Backward(int r, int c, double rg) const {
    const auto& elev = elevs.at(r);
    const auto& azim = azims.at(c);
    const auto r_xy = elev.cos * rg;
    return {r_xy * azim.cos, r_xy * azim.sin, rg * elev.sin};
  }
  cv::Point3d Backward(cv::Point px, double rg) const {
    return Backward(px.y, px.x, rg);
  }

  /// @brief Forward from xyzr to pixel
  cv::Point Forward(double x, double y, double z, double r) const {
    cv::Point px{-1, ToRowI(z, r)};
    // If row is oob then px.x will be -1 and will be bad
    if (px.y < 0 || px.y >= rows()) return px;
    // col is always good because 360
    px.x = ToColI(x, y);
    if (px.x >= cols()) px.x -= cols();
    return px;
  }
};

}  // namespace sv::rofl
