#pragma once

#include "sv/rofl/grid.h"
#include "sv/rofl/hess.h"
#include "sv/rofl/pwin.h"
#include "sv/rofl/traj.h"

namespace sv::rofl {

/// @brief Gicp config
struct GicpCfg {
  double stop_pos_tol{};      // max pos to early stop [m]
  double stop_rot_tol{};      // max rot to early stop [deg]
  int max_inner_iters{3};     // inner loop optimization
  int max_outer_iters{3};     // outer loop icp association
  int match_half_rows{2};     // pano match half rows
  int match_half_cols{2};     // pano match half cols
  bool use_all_panos{false};  // use only first pano in window

  const GicpCfg& Check() const;
  std::string Repr() const;

  /// @brief min_area = 0.5 * total_area + 1
  int min_area() const noexcept {
    return 2 * match_half_cols * match_half_rows + match_half_cols +
           match_half_rows + 2;
  }
  cv::Point half_px() const { return {match_half_cols, match_half_rows}; }
  cv::Size win_size2d() const {
    return {2 * match_half_cols + 1, 2 * match_half_rows + 1};
  }
};

struct GicpMatch {
  MeanCovar3f mc;        // mean and covar in pano frame
  cv::Point px{-1, -1};  // pano pixel
  double weight{0};      // cost weight
  int pano_id{-1};       // pano id in window

  void Reset() { pano_id = px.x = px.y = -1; }
  bool ok() const noexcept { return pano_id >= 0; }
  bool bad() const noexcept { return !ok(); }

  Eigen::Matrix3f CalcInfo(const Eigen::Matrix3f& cov_l,
                           const Eigen::Matrix3f& R_p_l) const;

  void UpdateHess(Hess1& hess,
                  const MeanCovar3f& mc_l,
                  const Eigen::Matrix3d& R_p_l,
                  const Eigen::Vector3d& x_pl) const;
};
using MatchGrid = Grid2d<GicpMatch>;

struct GicpStatus {
  std::string msg;
  double cost{};
  int num_panos{};
  int num_iters{};
  int num_costs{};
  bool ok{};

  std::string Repr() const;
};

/// @brief Generalized Icp
class GicpSolver {
  GicpCfg cfg_;
  cv::Mat pinds_;  // which pano in pwin has a match to (255 is no match)
  std::vector<SE3dVec> tfs_p_l_all_;
  MatchGrid matches_;  // grid of matches from sweep to pano
  int gsize_{};

  static constexpr uchar kBadPind = 255;

 public:
  explicit GicpSolver(const GicpCfg& cfg = {}) : cfg_{cfg.Check()} {}

  const MatchGrid& matches() const noexcept { return matches_; }
  const cv::Mat& pinds() const noexcept { return pinds_; }
  const auto& cfg() const noexcept { return cfg_; }
  std::string Repr() const;

  size_t Allocate(cv::Size grid_size);
  void Reset(cv::Range grid_span);

  /// @brief VICP wrt pano window
  GicpStatus Register(SweepGrid& grid,
                      Trajectory& traj,
                      const ImuQueue& imuq,
                      const PanoWindow& pwin,
                      const Projection& proj,
                      int gsize = 0);

 private:
  Hess1 BuildRigid(const SweepGrid& grid,
                   const PanoWindow& pwin,
                   const Projection& proj,
                   const Sophus::SE3d& dtf);
  Hess1 BuildRigid(const SweepGrid& grid,
                   const DepthPano& pano,
                   const Projection& proj,
                   const Sophus::SE3d& dtf,
                   int pind);

  /// @brief Precompute transforms from lidar (sweep) to pano, not thread-safe
  void CacheTransforms(const SE3fVec& tfs_o_l,
                       const Sophus::SE3d& tf_o_p,
                       const Sophus::SE3d& dtf_l,
                       SE3dVec& tfs_p_l);

  /// @brief Extract window to MeanCovar from pano
  /// @param mc mean and covar of pano window, will reset
  /// @param rg_l is range of lidar point projected to pano frame
  /// @return weight of this patch
  double ExtractPanoWin(MeanCovar3f& mc,
                        const Projection& proj,
                        const DepthPano& pano,
                        cv::Rect win,
                        double rg_l) const;

  /// @brief Check if we can stop early
  bool CheckEarlyStop(const Hess1::Vector6d& x, int scale) const;
};

}  // namespace sv::rofl
