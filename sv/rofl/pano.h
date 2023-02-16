#pragma once

#include <sophus/se3.hpp>

#include "sv/rofl/proj.h"
#include "sv/rofl/scan.h"

namespace sv::rofl {

struct PanoCfg {
  int max_info{10};  // max info per pixel
  double min_range{1.0};
  double max_range{100.0};
  double fuse_rel_tol{0.05};
  double fuse_abs_tol{0.5};

  const PanoCfg& Check() const;
  std::string Repr() const;

  /// @brief If rg is out of
  bool IsRangeBad(double rg) const { return rg < min_range || rg > max_range; }
};

struct PanoData {
  static constexpr auto kDtype = CV_16UC4;
  static constexpr double kRangeScale = 512.0;

  uint16_t r16u{};
  uint16_t s16u{};
  uint16_t info{};
  uint16_t g16u{};

  bool empty() const noexcept { return r16u == 0 && info == 0; }
  bool bad() const noexcept { return r16u == 0 || info == 0; }
  bool ok() const noexcept { return !bad(); }

  double GetRange() const noexcept {
    return static_cast<double>(r16u) / kRangeScale;
  }
  void SetRange(double rg) noexcept {
    r16u = static_cast<uint16_t>(rg * kRangeScale);
  }
  void SetSignal(double sg) noexcept { s16u = static_cast<uint16_t>(sg); }

  void DecInfo() noexcept {
    if (info > 0) --info;
  }
  void IncInfo(int max_info) noexcept {
    if (info < max_info) ++info;
  }
};

/// @brief Depth panorama described in "Mapping with depth panoramas"
class DepthPano final : public MatBase<PanoData> {
  PanoCfg cfg_;           // config
  int id_{-1};            // pano id
  double time_{0};        // creation time
  double num_sweeps_{0};  // num sweeps added to this pano
  Sophus::SE3d tf_o_p_;   // tf from pano to odom (fix odom frame, not imu!)
  SE3fVec tfs_p_l_;       // transform of each col to pano

 public:
  using DataT = PanoData;

  explicit DepthPano(cv::Size size = {1024, 256}, const PanoCfg& cfg = {});

  std::string Repr() const;
  int id() const noexcept { return id_; }
  double time() const noexcept { return time_; }
  const auto& cfg() const noexcept { return cfg_; }
  const auto& tf_o_p() const noexcept { return tf_o_p_; }
  double num_sweeps() const noexcept { return num_sweeps_; }

  void set_id(int id) noexcept { id_ = id; }

  /// @brief Allocate storage
  size_t Allocate(cv::Size pano_size) { return AllocateMat(pano_size); }

  /// @brief Reset this pano
  void Clear();
  void Reset(int id, const Sophus::SE3d& tf_o_p);

  /// @brief Render another pano into this pano
  /// @note Current pano must be an empty one
  int RenderPano(const DepthPano& pano1, const Projection& proj, int gsize = 0);

  /// @brief Add a sweep into this pano
  int AddSweep(const LidarSweep& sweep,
               const Projection& proj,
               cv::Range span,
               int gsize = 0);
  bool AddPoint(const Eigen::Vector3f& point,
                const Projection& proj,
                uint16_t s16u,
                uint16_t g16u);

  /// @brief Extrach single channel from panel
  void ExtractRange16U(cv::Mat& out) const;
  void ExtractCount16U(cv::Mat& out) const;
  void ExtractSignal16U(cv::Mat& out) const;
  void ExtractGrad16U(cv::Mat& out) const;
};

/// @brief Fuse
bool UpdateBuffer(PanoData& data, double range, uint16_t signal, int info);
bool Fuse(PanoData& data,
          const PanoCfg& cfg,
          double range,
          uint16_t signal,
          uint16_t grad);
DepthPano MakeTestPano(cv::Size size, double rg = 2.0, int info = 1);

}  // namespace sv::rofl
