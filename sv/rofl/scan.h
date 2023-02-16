#pragma once

#include <glog/logging.h>

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

namespace sv::rofl {

using SE3fVec = std::vector<Sophus::SE3f>;
using SE3dVec = std::vector<Sophus::SE3d>;

/// @brief Similar to c % cols but can only handle c \in [-cols, 2 * cols)
inline constexpr int WrapCols(int c, int cols) noexcept {
  return c < 0 ? c + cols : (c >= cols ? c - cols : c);
}

// TODO(rofl): change range_scale to 1 / range_scale
struct ScanInfo {
  uint64_t end_time_ns{}; // time of last col
  double col_dtime{};     // dt between two column
  float range_scale{};    // used to convert range from 16u to float
  cv::Range col_span{};   // col span

  void Check() const;
  std::string Repr() const;
};

/// @struct The layout should match that of the ouster decoder
struct ScanData {
  static constexpr int kDtype = CV_32FC4;

  float x{};
  float y{};
  float z{};
  uint16_t r16u{};  // range raw, need to be scaled
  uint16_t s16u{};  // signal

  std::string Repr() const;
  bool bad() const noexcept { return std::isnan(x); }
  bool ok() const noexcept { return !bad(); }
  auto xyz() const { return Eigen::Map<const Eigen::Vector3f>(&x); }
};
static_assert(sizeof(ScanData) == sizeof(float) * 4,
              "Size of ScanPixel must be 16 bytes");

/// @struct Base class to store scan data
template <typename T>
struct MatBase {
  cv::Mat mat_;

  MatBase() = default;
  MatBase(const cv::Mat& mat) : mat_{mat} {
    // Make sure T matches size of mat element
    CHECK_EQ(sizeof(T), mat.elemSize());
  }

  cv::Mat& mat() noexcept { return mat_; }
  const cv::Mat& mat() const noexcept { return mat_; }

  int rows() const noexcept { return mat_.rows; }
  int cols() const noexcept { return mat_.cols; }
  int type() const noexcept { return mat_.type(); }
  bool empty() const { return mat_.empty(); }
  size_t total() const { return mat_.total(); }
  cv::Size size2d() const { return {cols(), rows()}; }

  T& DataAt(int r, int c) { return mat_.at<T>(r, c); }
  T& DataAt(cv::Point px) { return mat_.at<T>(px); }

  const T& DataAt(int r, int c) const { return mat_.at<T>(r, c); }
  const T& DataAt(cv::Point px) const { return mat_.at<T>(px); }

  size_t AllocateMat(cv::Size size) {
    mat_.create(size, T::kDtype);
    return mat_.total() * mat_.elemSize();
  }
};

/// @brief LidarScan, this is just a temporary storage for scan.
struct LidarScan : public MatBase<ScanData> {
  ScanInfo info_;

  LidarScan() = default;
  LidarScan(const cv::Mat& mat, const ScanInfo& info);

  std::string Repr() const;

  const ScanInfo& info() const noexcept { return info_; }
  cv::Range span() const noexcept { return info_.col_span; }

  /// @brief Count number of valid points
  int CountNumValid(int gsize = 0) const;

  /// @brief Time at a particular column
  double TimeFromEnd(int n) const noexcept {
    return info_.end_time_ns/1.e9 - info_.col_dtime * n;
  }
  double TimeEnd() const noexcept { return TimeFromEnd(0); }
  double TimeEndNs() const noexcept { return info_.end_time_ns; }

  float RangeAt(int r, int c) const {
    return static_cast<float>(DataAt(r, c).r16u) / info_.range_scale;
  }
  float RangeAt(cv::Point px) const { return RangeAt(px.y, px.x); }

  /// @brief Extract range and signal
  void ExtractRange16u(cv::Mat& out) const;
  void ExtractSignal16u(cv::Mat& out) const;

  /// @brief Set this col_range to bad
  void SetBad(cv::Range& col_range);
};

/// @brief This buffer stores the full sweep, the score grid and matches, and
/// transforms of each column
class LidarSweep final : public LidarScan {
  cv::Mat ddrdu_;    // abs of 2nd order deriviative of range in x (float32)
  cv::Mat dsdu_;     // gradient of signal in x (uint16)
  SE3fVec tfs_o_l_;  // transform from each lidar column to odom frame

 public:
  LidarSweep() = default;
  explicit LidarSweep(cv::Size size) { Allocate(size); }

  const cv::Mat& ddrdu() const noexcept { return ddrdu_; }
  const cv::Mat& dsdu() const noexcept { return dsdu_; }
  //  const cv::Mat& edge() const noexcept { return edge_; }

  auto& tfs() noexcept { return tfs_o_l_; }
  const auto& tfs() const noexcept { return tfs_o_l_; }
  const auto& TfAt(int c) const { return tfs_o_l_.at(c); }

  /// @brief Whether this sweep is taking partial sweep input
  bool IsPartial() const { return span().size() < cols(); }

  /// @brief Allocate storage
  size_t Allocate(cv::Size size);

  /// @brief Add scan to sweep buffer
  /// @details Normally the start col of the new scan should match the end col
  /// of the previous scan in sweep. But if we miss any data in between, we will
  /// set the skipped cols invalid.
  /// @return Number of pixels added (including invalid)
  size_t Add(const LidarScan& scan);

  /// @brief Fill small holes in partial sweep
  /// @return Number of holes filled
  int FillHoles(int gsize = 0);

  /// @group Compute 1st/2nd order gradient of range/signal
  void CalcRangeGrad2(int gsize = 0);
  void CalcSignalGrad(int gsize = 0);
  void CalcSignalEdge(int min_val, int gsize = 0);

  /// @brief Return valid range such that we don't need to check bounds
  cv::Range GetValidRange(int border) const;

 private:
  /// @brief Set sweep info using scan info
  void SetInfo(const ScanInfo& info);
};

/// @brief Test helper functions
cv::Mat MakeTestScanMat(cv::Size size, float range = 2.0F);
LidarScan MakeTestScan(cv::Size size, float range = 2.0F);
LidarSweep MakeTestSweep(cv::Size size, float range = 2.0F);

}  // namespace sv::rofl
