#include "sv/rofl/pano.h"

#include <opencv2/core.hpp>  // extractChannel

#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::rofl {

namespace {

/// @brief Make Gsize at least 2
int GsizeAtLeast2(int gsize) { return gsize == 1 ? 2 : gsize; }

}  // namespace

using SE3d = Sophus::SE3d;
using SE3f = Sophus::SE3f;
using SO3d = Sophus::SO3d;
using Vector3f = Eigen::Vector3f;
using Vector3d = Eigen::Vector3d;

const PanoCfg& PanoCfg::Check() const {
  CHECK_GT(max_info, 0);
  CHECK_GT(min_range, 0);
  CHECK_GT(max_range, 0);
  CHECK_LT(min_range, max_range);
  CHECK_LT(max_range * PanoData::kRangeScale,
           std::numeric_limits<uint16_t>::max());
  return *this;
}

std::string PanoCfg::Repr() const {
  return fmt::format(
      "PanoCfg(max_info={}, min_range={}, max_range={}, fuse_rel_tol={}, "
      "fuse_abs_tol={})",
      max_info,
      min_range,
      max_range,
      fuse_rel_tol,
      fuse_abs_tol);
}

DepthPano::DepthPano(cv::Size size, const PanoCfg& cfg) : cfg_{cfg.Check()} {
  Allocate(size);
  mat_.setTo(0);
}

std::string DepthPano::Repr() const {
  return fmt::format("DepthPano(id={}, size={}x{}, num_sweeps={}, cfg={})",
                     id(),
                     rows(),
                     cols(),
                     num_sweeps_,
                     cfg_.Repr());
}

void DepthPano::Clear() {
  id_ = -1;
  num_sweeps_ = 0;
  mat_.setTo(0);
}

void sv::rofl::DepthPano::Reset(int id, const Sophus::SE3d& tf_o_p) {
  Clear();
  id_ = id;
  tf_o_p_ = tf_o_p;
}

int DepthPano::RenderPano(const DepthPano& pano1,
                          const Projection& proj,
                          int gsize) {
  CHECK(!empty());
  CHECK(!proj.empty());
  CHECK_EQ(rows(), proj.rows());
  CHECK_EQ(cols(), proj.cols());

  // make sure pano1 is good
  CHECK_GE(pano1.id(), 0);
  CHECK_GT(pano1.num_sweeps(), 1);

  // This must be an empty pano
  CHECK_EQ(num_sweeps_, 0);
  // Bump num_sweeps by 1
  num_sweeps_++;

  // new pano is pano1, this pano is pano0
  // tf_0_1 = tf_0_o * tf_o_1
  const auto tf_0_1 = tf_o_p_.inverse() * pano1.tf_o_p();

  return ParallelReduce(
      {0, rows(), GsizeAtLeast2(gsize)},
      0,
      [&](int pr, int& n) {
        for (int pc = 0; pc < cols(); ++pc) {
          const auto& data1 = pano1.DataAt(pr, pc);

          // Only fuse point that are sufficiently converged
          if (data1.r16u == 0 || data1.info < cfg_.max_info / 2) continue;

          // Convert to 3d point in pano1 frame
          const auto pt1 = proj.Backward(pr, pc, data1.GetRange());
          Eigen::Map<const Vector3d> pt1_map(&pt1.x);

          // Transform to pano0 frame (this pano)
          const Vector3d pt0 = tf_0_1 * pt1_map;

          // Compute range and check
          const auto rg0 = pt0.norm();
          CHECK(!std::isnan(rg0));
          if (cfg_.IsRangeBad(rg0)) continue;

          // Project to current pano pixel and check
          const auto px0 = proj.Forward(pt0.x(), pt0.y(), pt0.z(), rg0);
          if (IsPixBad(px0)) continue;

          // Update depth buffer (not using Fuse() because we know we this pano
          // is empty)
          n += UpdateBuffer(DataAt(px0), rg0, data1.s16u, data1.info);
        }
      },
      std::plus<>{});
}

int DepthPano::AddSweep(const LidarSweep& sweep,
                        const Projection& proj,
                        cv::Range span,
                        int gsize) {
  CHECK(!empty());
  CHECK(!proj.empty());
  CHECK_EQ(rows(), proj.rows());
  CHECK_EQ(cols(), proj.cols());

  // update num_sweeps
  num_sweeps_ += static_cast<double>(span.size()) / sweep.cols();

  // Precompute transform from lidar to pano
  // Because tfs in sweep are from local to odom
  tfs_p_l_.resize(span.size());
  const SE3f tf_p_o = tf_o_p_.inverse().cast<float>();
  for (int sc = span.start; sc < span.end; ++sc) {
    // tf_p_l = tf_p_o * tf_o_l
    tfs_p_l_.at(sc - span.start) = tf_p_o * sweep.tfs().at(sc);
  }

  // Make sure minimum gsize is 2, this is a hack to reduce data race
  return ParallelReduce(
      {0, sweep.rows(), GsizeAtLeast2(gsize)},
      0,
      [&](int sr, int& n) {
        for (int sc = span.start; sc < span.end; ++sc) {
          const auto& data = sweep.DataAt(sr, sc);
          if (data.bad()) {
            // TODO (rofl): maybe do something when data is bad
          } else {
            // Transform point into pano frame
            const auto& tf_p_l = tfs_p_l_.at(sc - span.start);
            // pt_pl = T_p_l * pt_l
            const auto pt = tf_p_l * data.xyz();
            // TODO (rofl): use gradient?
            const auto dsdu = sweep.dsdu().at<uint16_t>(sr, sc);
            n += AddPoint(pt, proj, data.s16u, dsdu);
          }
        }
      },
      std::plus<>{});
}

bool DepthPano::AddPoint(const Vector3f& pt,
                         const Projection& proj,
                         uint16_t s16u,
                         uint16_t g16u) {
  // Transform into pano frame
  const auto rg = pt.norm();  // range in pano frame

  // Ignore too far and too close point
  CHECK(!std::isnan(rg));
  if (cfg_.IsRangeBad(rg)) return false;

  // Project to pano
  const auto px = proj.Forward(pt.x(), pt.y(), pt.z(), rg);
  if (IsPixBad(px)) return false;

  // TODO (rofl): should we use different info based on distance to center?
  // Fuse with current pano pixel
  return Fuse(DataAt(px), cfg_, rg, s16u, g16u);
}

void DepthPano::ExtractRange16U(cv::Mat& out) const {
  cv::extractChannel(mat_, out, 0);
}

void DepthPano::ExtractCount16U(cv::Mat& out) const {
  cv::extractChannel(mat_, out, 2);
}

void DepthPano::ExtractSignal16U(cv::Mat& out) const {
  cv::extractChannel(mat_, out, 1);
}

void DepthPano::ExtractGrad16U(cv::Mat& out) const {
  cv::extractChannel(mat_, out, 3);
}

bool Fuse(PanoData& data,
          const PanoCfg& cfg,
          double range,
          uint16_t signal,
          uint16_t grad) {
  // If depth is 0, this is a new point
  if (data.empty()) {
    data.SetRange(range);
    data.s16u = signal;
    data.info = static_cast<uint16_t>(cfg.max_info / 2);
    return true;
  }

  // Otherwise we have a vlid depth with some evidence (info > 0)
  const auto range_old = data.GetRange();
  const auto range_abs_diff = std::abs(range - range_old);

  // This is a bad point, only decrement info
  if (range_abs_diff > cfg.fuse_abs_tol ||
      (range_abs_diff / range_old) > cfg.fuse_rel_tol) {
    data.DecInfo();
    return false;
  }

  const double denom = data.info + 1.0;

  // close enough, do weighted update
  const auto rg_fuse = (range_old * data.info + range) / denom;
  data.SetRange(rg_fuse);

  // Update signal and grad
  data.s16u = static_cast<uint16_t>((data.s16u * data.info + signal) / denom);
  data.g16u = static_cast<uint16_t>((data.g16u * data.info + grad) / denom);

  // Update info
  data.IncInfo(cfg.max_info);

  return true;
}

DepthPano MakeTestPano(cv::Size size, double rg, int info) {
  DepthPano pano(size);

  CHECK_LE(info, pano.cfg().max_info);

  for (int r = 0; r < pano.rows(); ++r) {
    for (int c = 0; c < pano.cols(); ++c) {
      auto& data = pano.DataAt(r, c);
      data.SetRange(rg);
      data.info = static_cast<uint16_t>(info);
      data.s16u = 512;
    }
  }

  return pano;
}

bool UpdateBuffer(PanoData& data, double range, uint16_t signal, int info) {
  if (data.r16u == 0 || range < data.GetRange()) {
    // When rendering a new depth pano, if the original pixel is well estimated
    // (high cnt), this means that it also has good visibility from the current
    // viewpoint. On the other hand, if it has low cnt, this means that it was
    // probably occluded. Therefore, we simply half the original cnt and make it
    // the new one
    data.SetRange(range);
    data.info = info / 2;
    data.s16u = signal;
    return true;
  }
  return false;
}

}  // namespace sv::rofl
