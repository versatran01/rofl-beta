#include "sv/rofl/grid.h"

#include "sv/util/logging.h"
#include "sv/util/opencv.h"
#include "sv/util/tbb.h"

namespace sv::rofl {

const GridCfg& GridCfg::Check() const {
  CHECK_GT(cell_rows, 0);
  CHECK_GE(cell_cols, 8);
  CHECK_GE(feat_min_range, 0);
  // Due to hole filling there might be one invalid pixel, so we require a min
  // width of 4 to ensure we have at least 3 points
  CHECK_GE(feat_min_pixels, 4);
  CHECK_GT(feat_max_smooth, 0);
  CHECK(IsPowerOf2(cell_cols)) << cell_cols << " is not power of 2";
  return *this;
}

std::string GridCfg::Repr() const {
  return fmt::format(
      "GridCfg(cell_size={}x{}, sel_max_smooth={}, sel_min_pixels={}, "
      "sel_min_length={}, sel_nms_dist={})",
      cell_rows,
      cell_cols,
      feat_max_smooth,
      feat_min_pixels,
      feat_min_length,
      feat_nms_dist);
}

std::string SweepGrid::Repr() const {
  return fmt::format(
      "SweepGrid(grid={}x{}, cfg={})", rows(), cols(), cfg_.Repr());
}

size_t SweepGrid::Allocate(cv::Size sweep_size) {
  // Make sure sweep size is divisible by cell size
  CHECK_EQ(sweep_size.height % cfg_.cell_rows, 0);
  CHECK_EQ(sweep_size.width % cfg_.cell_cols, 0);

  const auto grid_size = sweep_size / cell_size2d();
  points_.resize(grid_size);
  tfs_o_l_.resize(grid_size.width);

  return points_.size() * sizeof(GridPoint) +
         tfs_o_l_.size() * sizeof(Sophus::SE3f);
}

int SweepGrid::Select(const LidarSweep& sweep, int gsize) {
  CHECK_EQ(rows() * cfg_.cell_rows, sweep.rows());
  CHECK_EQ(cols() * cfg_.cell_cols, sweep.cols());

  // Update grid span with sweep span
  span_ = sweep.span() / cfg_.cell_cols;

  // Precompute some useful information used during selection
  const auto delta_azimuth = M_PI * 2 / sweep.cols();
  // Given min_length what is the required cols at a particular range?
  const auto length_over_theta = cfg_.feat_min_length / delta_azimuth;
  const auto nms_dist_sq = cfg_.feat_nms_dist * cfg_.feat_nms_dist;

  VLOG(10) << fmt::format(
      "[Select] span=[{},{}), delta_azimuth={:.3f}, length_over_theta={:.3f}",
      span_.start,
      span_.end,
      delta_azimuth,
      length_over_theta);

  return ParallelReduce(
      {0, rows(), gsize},
      0,
      [&](int gr, int& n) {
        for (int gc = span_.start; gc < span_.end; ++gc) {
          auto& point = points_.at(gr, gc);
          point.Reset();

          // cell rect in sweep
          const cv::Rect rect{{gc * cfg_.cell_cols, gr * cfg_.cell_rows},
                              cell_size2d()};
          // Select the longest flat area from this cell (rowwise)
          // If not found then we get an invalid xyw (w=0) anyway
          const auto xyw = SelectFrom(
              sweep.ddrdu(), rect, static_cast<float>(cfg_.feat_max_smooth));

          // 0. must have at least min_width
          if (xyw.w < cfg_.feat_min_pixels) continue;

          // 1. must not be too close
          const auto px_mid = xyw.px_mid();
          const auto rg_mid = sweep.RangeAt(px_mid);  // range of mid pixel
          // a bad point wil have rg 0
          if (rg_mid <= cfg_.feat_min_range) continue;

          // 2. Compute required cols based on range
          // Such that points span at least min_length
          // min_cols = min_length / delta_azimuth / range
          const int min_cols = static_cast<int>(length_over_theta / rg_mid);
          if (xyw.w < std::min(cfg_.cell_cols - 1, min_cols)) continue;

          // Do nms based on distance, only look to the left (for now)
          if (cfg_.feat_nms_dist > 0 && gc > 0) {
            const auto& data_mid = sweep.DataAt(px_mid);
            const auto& pt_left = points_.at(gr, gc - 1);
            // Skip if left is selected and distance is too close
            if (pt_left.ok() &&
                (pt_left.mc.mean - data_mid.xyz()).squaredNorm() < nms_dist_sq)
              continue;
          }

          // Otherwise we keep it and compute mean and covar
          point.xyw = xyw;
          CalcMeanCovar(sweep, point.xyw, point.mc);
          if (!point.ok()) continue;
          //CHECK(point.ok()) << point.Repr();

          ++n;
        }  // gc
      },   // gr
      std::plus<>{});
}

void SweepGrid::MakeSelectMask(cv::Mat& mask) const {
  mask.create(cfg_.cell_rows * rows(), cfg_.cell_cols * cols(), CV_8UC1);
  mask.setTo(0);

  for (int gr = 0; gr < rows(); ++gr) {
    for (int gc = 0; gc < cols(); ++gc) {
      const auto& point = points_.at(gr, gc);
      if (point.xyw.w == 0) continue;
      MatSetRoi<uint8_t>(mask, {point.px(), cv::Size{point.width(), 1}}, 255);
    }
  }
}

int SweepGrid::GetNumSelected() const {
  int n{};
  for (const auto& point : points_) n += point.ok();
  return n;
}

cv::Range FindLongestRange(const VectorXfCRef& ddr, float max_ddr) {
  const auto cols = ddr.size();

  cv::Range rg{};
  // 1st ptr
  int i = 0;

  while (i < cols) {
    // skip anything that is bad (ddr[i] could be nan or inf)
    if (!(ddr[i] <= max_ddr)) {
      ++i;
      continue;
    }

    // 2nd ptr
    auto j = i + 1;

    // keep moving if element is good
    while (j < cols && ddr[j] <= max_ddr) ++j;

    if (j - i > rg.size()) {
      rg.start = i;
      rg.end = j;
    }

    // update 1st ptr
    i = j + 1;
  }

  return rg;
}

PixelWidth SelectFrom(const cv::Mat& ddrdu,
                      const cv::Rect& rect,
                      float max_ddr) {
  PixelWidth xyw;

  // For each row in cell
  for (int y = rect.y; y < rect.y + rect.height; ++y) {
    // Find the longest flat area in this row
    // Map this row in cell
    VectorXfCMap cell_row_map(ddrdu.ptr<float>(y, rect.x), rect.width);
    const auto rg = FindLongestRange(cell_row_map, max_ddr);
    // Update best feat
    if (rg.size() > xyw.w) {
      xyw.x = rg.start + rect.x;
      xyw.y = y;
      xyw.w = rg.size();
    }
  }

  // If not found we return default
  return xyw;
}

void CalcMeanCovar(const LidarSweep& sweep,
                   const PixelWidth& xyw,
                   MeanCovar3f& mc) {
  mc.Reset();
  for (int c = 0; c < xyw.w; ++c) {
    const auto& data = sweep.DataAt(xyw.y, xyw.x + c);
    // Due to hole filling we might have a bad point somewhere
    if (data.bad()) continue;
    // Otherwise add to point
    mc.Add(data.xyz());
  }
}

std::string GridPoint::Repr() const {
  return fmt::format(
      "GridPoint(x={}, y={}, w={}, n={}", xyw.x, xyw.y, xyw.w, mc.n);
}

}  // namespace sv::rofl
