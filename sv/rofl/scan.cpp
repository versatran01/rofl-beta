#include "sv/rofl/scan.h"

#include <opencv2/core.hpp>  // extract channel

#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::rofl {

namespace {
constexpr auto kNaNF = std::numeric_limits<float>::quiet_NaN();
template <typename T>
constexpr T Sq(T x) noexcept {
  return x * x;
}
}  // namespace

void ScanInfo::Check() const {
  CHECK_GE(end_time_ns, 0);
  CHECK_GE(col_dtime, 0);
  CHECK_GE(col_span.start, 0);
}

std::string ScanInfo::Repr() const {
  return fmt::format(
      "col_span=[{},{}), end_time={}, col_dt={}ns, range_scale={}",
      col_span.start,
      col_span.end,
      static_cast<double>(end_time_ns) / 1e9,
      static_cast<int>(col_dtime * 1e9),
      range_scale);
}

std::string ScanData::Repr() const {
  return fmt::format("ScanData(xyz=[{:.3f}, {:.3f}, {:.3f}], r={}, s={})",
                     x,
                     y,
                     z,
                     r16u,
                     s16u);
}

/// ============================================================================
LidarScan::LidarScan(const cv::Mat& mat, const ScanInfo& info)
    : MatBase{mat}, info_{info} {
  info_.Check();
  CHECK_EQ(info_.col_span.size(), mat.cols);
  CHECK(!mat.empty());
}

std::string LidarScan::Repr() const {
  return fmt::format("LidarScan(size={}x{}, dtype={}, {})",
                     cols(),
                     rows(),
                     type(),
                     info_.Repr());
}

int LidarScan::CountNumValid(int gsize) const {
  return ParallelReduce(
      {0, rows(), gsize},
      0,
      [&](int r, int& n) {
        for (int c = 0; c < cols(); ++c) {
          n += static_cast<int>(DataAt(r, c).ok());
        }
      },
      std::plus<>{});
}

void LidarScan::ExtractRange16u(cv::Mat& out) const {
  CHECK(!empty());
  cv::Mat map(size2d(), CV_16UC(8), mat_.data);
  cv::extractChannel(map, out, 6);
}

void LidarScan::ExtractSignal16u(cv::Mat& out) const {
  CHECK(!empty());
  cv::Mat map(size2d(), CV_16UC(8), mat_.data);
  cv::extractChannel(map, out, 7);
}

void LidarScan::SetBad(cv::Range& col_range) {
  CHECK_GE(col_range.start, 0);
  CHECK_LT(col_range.end, cols());
  if (col_range.empty()) return;

  // Note: this is a bit unsafe, because the last two channels (range, signal)
  // will be (0, 32704).
  mat_.colRange(col_range).setTo(kNaNF);
}

/// ============================================================================
size_t LidarSweep::Allocate(cv::Size size) {
  constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
  ddrdu_.create(size, CV_32FC1);
  ddrdu_.setTo(nan);

  dsdu_.create(size, CV_16UC1);
  dsdu_.setTo(0);

  //  edge_.create(size, CV_16SC1);
  //  edge_.setTo(0);

  tfs_o_l_.resize(size.width);

  const auto mat_bytes = AllocateMat(size);
  return ddrdu_.total() * ddrdu_.elemSize() + dsdu_.total() * dsdu_.elemSize() +
         // edge_.total() * edge_.elemSize() +
         tfs_o_l_.size() * sizeof(Sophus::SE3f) + mat_bytes;
}

size_t LidarSweep::Add(const LidarScan& scan) {
  CHECK(!empty()) << "Sweep is not allocated";
  CHECK_EQ(type(), scan.type()) << "Sweep/Scan type mismatch";
  CHECK_EQ(rows(), scan.rows()) << "Sweep/Scan row mismatch";
  CHECK_GE(cols(), scan.cols()) << "Scan has too many cols";

  SetInfo(scan.info());
  scan.mat().copyTo(mat_.colRange(scan.span()));  // copy data
  return scan.total();
}

void LidarSweep::SetInfo(const ScanInfo& new_info) {
  // Make sure range_scale and col_dt are the same
  if (info_.range_scale > 0) CHECK_EQ(info_.range_scale, new_info.range_scale);
  info_.range_scale = new_info.range_scale;

  if (info_.col_dtime > 0) CHECK_EQ(info_.col_dtime, new_info.col_dtime);
  info_.col_dtime = new_info.col_dtime;

  // Make sure time only goes forward
  if (info_.end_time_ns > 0) CHECK_LT(info_.end_time_ns, new_info.end_time_ns);
  info_.end_time_ns = new_info.end_time_ns;

  // TODO (rofl): need to return jumped cols
  info_.col_span = new_info.col_span;
}

int LidarSweep::FillHoles(int gsize) {
  // offset by 1 such that we don't need to check bounds
  const auto col_range = GetValidRange(1);

  return ParallelReduce(
      {0, rows(), gsize},
      0,
      [&](int sr, int& n) {
        for (int sc = col_range.start; sc < col_range.end; ++sc) {
          auto& xm = DataAt(sr, sc);
          if (xm.r16u > 0) continue;

          const auto& xl = DataAt(sr, sc - 1);
          const auto& xr = DataAt(sr, sc + 1);
          // If any of left or right is bad, then don't fill
          if (xl.r16u == 0 || xr.r16u == 0) continue;

          // Fill both range and signal
          xm.r16u = xl.r16u / 2 + xr.r16u / 2;
          xm.s16u = xl.s16u / 2 + xr.s16u / 2;

          ++sc;  // skip next
          ++n;
        }  // c
      },   // r
      std::plus<>{});
}

cv::Range LidarSweep::GetValidRange(int border) const {
  CHECK_GE(border, 0);

  int col_beg = span().start;
  if (col_beg == 0) col_beg += border;

  int col_end = span().end;
  if (col_end == cols()) col_end -= border;

  return {col_beg, col_end};
}

void LidarSweep::CalcRangeGrad2(int gsize) {
  // Skip first and last to avoid bound checks (they will be nan)
  const auto col_range = GetValidRange(1);

  // We hard code max range to 100m, this is used to scale denominator
  constexpr float r_max_inv = 1.0F / 100.0F;

  ParallelFor({0, rows(), gsize}, [&](int r) {
    for (int c = col_range.start; c < col_range.end; ++c) {
      auto& ddr = ddrdu_.at<float>(r, c);
      const float r16u_m = DataAt(r, c).r16u;

      // If this pixel is bad, set ddr to nan
      if (r16u_m == 0) {
        ddr = kNaNF;
        continue;
      }

      // Computes 2nd order derivative of range image
      // NOTE: we don't check whether neighboring pixels are valid (>0), because
      // if they are not (=0), it will result in a large value anyway and will
      // be filtered out later during selection.
      // ddr is computed as
      // ddr = (left + right - mid * 2) / ((mid / max + 1) * 5)
      // the denominator is a heuristic to scale the score, to allow more far
      // points to be selected
      const float r16u_l = DataAt(r, c - 1).r16u;
      const float r16u_r = DataAt(r, c + 1).r16u;
      const float denom = r16u_m * r_max_inv + info_.range_scale;
      ddr = std::abs((r16u_l + r16u_r - r16u_m - r16u_m) / denom);
    }  // sc
  });  // sr
}

void LidarSweep::CalcSignalGrad(int gsize) {
  using elem_t = uint16_t;

  const auto col_range = GetValidRange(2);

  ParallelFor({0, rows(), gsize}, [&](int r) {
    auto calc = [&](int col) -> float {
      const auto& data = DataAt(r, col);
      return Sq(data.r16u / info_.range_scale) * data.s16u / 100.0F;
    };

    for (int c = col_range.start; c < col_range.end; ++c) {
      auto& g = dsdu_.at<elem_t>(r, c);
      const auto d = DataAt(r, c);

      // Skip if current pixel is bad
      if (d.bad()) {
        g = 0;
        continue;
      }

      const auto s_l2 = calc(c - 2);
      const auto s_l1 = calc(c - 1);
      const auto s_r1 = calc(c + 1);
      const auto s_r2 = calc(c + 2);

      // const int s_l2 = DataAt(r, c - 2).r16u;
      // const int s_l1 = DataAt(r, c - 1).r16u;
      // const int s_r1 = DataAt(r, c + 1).r16u;
      // const int s_r2 = DataAt(r, c + 2).r16u;

      // Also skip if any neighboring pixel is bad
      if (s_l2 == 0 || s_l1 == 0 || s_r1 == 0 || s_r2 == 0) {
        g = 0;
        continue;
      }

      g = static_cast<elem_t>(std::abs(s_r2 + s_r1 - s_l1 - s_l2) / 4);
    }
  });
}

// void LidarSweep::CalcSignalEdge(int min_ds, int gsize) {
//   CHECK_GT(min_ds, 0);

//  using elem_t = int16_t;
//  // kernel is [1, 4, 6, 4, 1]
//  const auto col_range = GetValidRange(2);

//  ParallelFor({0, rows(), gsize}, [&](int r) {
//    for (int c = col_range.start; c < col_range.end; ++c) {
//      const auto x0 = dsdu_.at<elem_t>(r, c - 1);
//      const auto x1 = dsdu_.at<elem_t>(r, c);
//      const auto x2 = dsdu_.at<elem_t>(r, c + 1);

//      auto row_ptr = edge_.ptr<elem_t>(r);

//      if (x1 > min_ds && x1 > x0 && x1 > x2) {
//        row_ptr[c - 2] = 1;
//        row_ptr[c - 1] = 4;
//        row_ptr[c] = 6;
//        row_ptr[c + 1] = 4;
//        row_ptr[c + 2] = 1;
//      } else if (x1 < -min_ds && x1 < x0 && x1 < x2) {
//        row_ptr[c - 2] = -1;
//        row_ptr[c - 1] = -4;
//        row_ptr[c] = -6;
//        row_ptr[c + 1] = -4;
//        row_ptr[c + 2] = -1;
//      } else {
//        row_ptr[c - 2] = 0;
//        row_ptr[c - 1] = 0;
//        row_ptr[c] = 0;
//        row_ptr[c + 1] = 0;
//        row_ptr[c + 2] = 0;
//      }
//    }
//  });
//}

/// ============================================================================
cv::Mat MakeTestScanMat(cv::Size size, float range) {
  cv::Mat data;
  data.create(size, ScanData::kDtype);
  constexpr auto pi = static_cast<float>(M_PI);

  const float azim_delta = pi * 2.0F / static_cast<float>(size.width);
  const float elev_max = pi / 4.0F;
  const float elev_delta =
      elev_max * 2.0F / static_cast<float>((size.height - 1));

  for (int i = 0; i < data.rows; ++i) {
    for (int j = 0; j < data.cols; ++j) {
      const float elev = elev_max - i * elev_delta;
      const float azim = pi * 2.0F - j * azim_delta;

      auto& p = data.at<ScanData>(i, j);
      p.x = std::cos(elev) * std::cos(azim) * range;
      p.y = std::cos(elev) * std::sin(azim) * range;
      p.z = std::sin(elev) * range;
      p.r16u = static_cast<uint16_t>(512 * range);
      p.s16u = static_cast<uint16_t>(512);
    }
  }
  return data;
}

LidarScan MakeTestScan(cv::Size size, float range) {
  const auto mat = MakeTestScanMat(size, range);
  ScanInfo info;
  info.end_time_ns = 1;
  info.col_dtime = 1.0 / size.width;
  info.range_scale = 512.0F;
  info.col_span = {0, size.width};
  return {mat, info};
}

LidarSweep MakeTestSweep(cv::Size size, float range) {
  LidarSweep sweep(size);
  sweep.Add(MakeTestScan(size, range));
  return sweep;
}

}  // namespace sv::rofl
