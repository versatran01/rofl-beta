#pragma once

#include "sv/rofl/scan.h"    // LidarSweep
#include "sv/util/eigen.h"   // VectorXfCRef
#include "sv/util/grid2d.h"  // Grid2d
#include "sv/util/math.h"    // MeanCovar

namespace sv::rofl {

/// @brief Stores starting pix coord and width in one scan line
struct PixelWidth {
  int x{};
  int y{};
  int w{};

  cv::Point px() const noexcept { return {x, y}; }
  cv::Point px_mid() const noexcept { return {x + w / 2, y}; }
};

/// @brief Grid point
/// @todo (rofl) this will probably change when we add intensity
struct GridPoint {
  PixelWidth xyw{};  // 12
  MeanCovar3f mc{};  // 52 mean, covar

  auto width() const noexcept { return xyw.w; }
  auto px() const noexcept { return xyw.px(); }

  bool ok() const noexcept { return xyw.w > 0 && mc.n >= 3; }
  bool bad() const noexcept { return !ok(); }
  void Reset() { xyw = {}; }

  std::string Repr() const;
};
static_assert(sizeof(GridPoint) == 64);  // one cacheline
using PointGrid = Grid2d<GridPoint>;

struct GridCfg {
  double feat_max_smooth{0.1};  // max smoothness score
  double feat_min_length{0.5};  // min arc length for good point
  double feat_min_range{1.0};   // min range of center
  double feat_nms_dist{1.0};    // non-max suppresion distance
  int feat_min_pixels{4};       // min pixels for good cell
  int cell_rows{1};
  int cell_cols{16};

  const GridCfg& Check() const;
  std::string Repr() const;
};

class SweepGrid {
  GridCfg cfg_;
  cv::Range span_;    // grid span (matches sweep span)
  PointGrid points_;  // grid points
  SE3fVec tfs_o_l_;   // from lidar to odom

 public:
  explicit SweepGrid(const GridCfg& cfg = {}) : cfg_{cfg.Check()} {}

  std::string Repr() const;
  const auto& cfg() const noexcept { return cfg_; }
  const PointGrid& points() const noexcept { return points_; }
  const GridPoint& PointAt(int gr, int gc) const { return points_.at(gr, gc); }

  auto& tfs() noexcept { return tfs_o_l_; }
  const auto& tfs() const noexcept { return tfs_o_l_; }
  const auto& TfAt(int c) const { return tfs_o_l_.at(c); }

  size_t Allocate(cv::Size sweep_size);

  /// @brief Select pixels from the current span in sweep and reduce to point
  /// @details The sweep is divided into a grid of cells. Within each cell, we
  /// find the longest flat pixels that has 2nd order range derivative smaller
  /// than max_g2. We keep results based on the cfg.
  /// @return number select pixels
  int Select(const LidarSweep& sweep, int gsize = 0);

  /// @brief Make selection mask (for visualization)
  void MakeSelectMask(cv::Mat& mask) const;
  /// @brief Number of selected points
  int GetNumSelected() const;

  cv::Range span() const noexcept { return span_; }
  int rows() const noexcept { return points_.rows(); }
  int cols() const noexcept { return points_.cols(); }
  size_t size() const noexcept { return points_.size(); }
  cv::Size size2d() const noexcept { return points_.size2d(); }
  cv::Size cell_size2d() const { return {cfg_.cell_cols, cfg_.cell_rows}; }
};

/// @brief Select cell from rect in d2rdu
PixelWidth SelectFrom(const cv::Mat& ddr, const cv::Rect& rect, float max_ddr);

/// @brief Find longest consecutive flat area
/// @return [start_col, end_col)
cv::Range FindLongestRange(const VectorXfCRef& ddr, float max_ddr);

/// @brief Calc mean and covar for points in xyw from sweep
void CalcMeanCovar(const LidarSweep& sweep,
                   const PixelWidth& xyw,
                   MeanCovar3f& mc);

}  // namespace sv::rofl
