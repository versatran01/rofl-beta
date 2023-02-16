#include "sv/rofl/gicp.h"

#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::rofl {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using SE3f = Sophus::SE3f;
using Vector3d = Eigen::Vector3d;
using Vector3f = Eigen::Vector3f;
using Matrix3d = Eigen::Matrix3d;
using Matrix3f = Eigen::Matrix3f;

namespace {

// TODO (rofl): move this to proj?
cv::Point ProjWithBorder(const Projection& proj,
                         const Vector3d& pt,
                         double rg,
                         cv::Point border) {
  cv::Point px{-1, -1};

  const auto y = proj.ToRowI(pt.z(), rg);
  if (y < border.y || y >= proj.rows() - border.y) return px;

  const auto x = proj.ToColI(pt.x(), pt.y());
  if (x < border.x || x >= proj.cols() - border.x) return px;

  px = {x, y};
  return px;
}

}  // namespace

/// ============================================================================
const GicpCfg& GicpCfg::Check() const {
  CHECK_GE(stop_pos_tol, 0);
  CHECK_GE(stop_rot_tol, 0);
  CHECK_GT(max_inner_iters, 0);
  CHECK_GT(max_outer_iters, 0);
  CHECK_GT(match_half_rows, 1);
  CHECK_GT(match_half_cols, 1);
  return *this;
}

std::string GicpCfg::Repr() const {
  return fmt::format(
      "GicpCfg(max_inner_iters={}, max_outer_iters={}, "
      "match_half_rows={}, match_half_cols={}, "
      "stop_pos_tol={}, stop_rot_tol={}, "
      "use_all_panos={})",
      max_inner_iters,
      max_outer_iters,
      match_half_rows,
      match_half_cols,
      stop_pos_tol,
      stop_rot_tol,
      use_all_panos);
}

/// ============================================================================
std::string GicpStatus::Repr() const {
  return fmt::format(
      "GicpStatus(num_panos={}, num_iters={}, num_costs={}, costs={:.3f}, "
      "ok={}, msg={})",
      num_panos,
      num_iters,
      num_costs,
      cost,
      ok,
      msg);
}

/// ============================================================================
Matrix3f GicpMatch::CalcInfo(const Matrix3f& cov_l,
                             const Matrix3f& R_p_l) const {
  auto cov = mc.Covar();
  cov.noalias() += R_p_l * cov_l * R_p_l.transpose();
  return cov.inverse() * weight;
}

void GicpMatch::UpdateHess(Hess1& hess,
                           const MeanCovar3f& mc_l,
                           const Matrix3d& R_p_l,
                           const Vector3d& x_pl) const {
  const Matrix3f W = CalcInfo(mc_l.Covar(), R_p_l.cast<float>());
  const Vector3d r = mc.mean.cast<double>() - x_pl;

  Hess1::Matrix36d J;
  J.leftCols<3>().noalias() = R_p_l * Hat3(mc_l.mean).cast<double>();
  J.rightCols<3>() = -R_p_l;
  hess.Add(J, W.cast<double>(), r);
}

/// ============================================================================
std::string GicpSolver::Repr() const {
  return fmt::format("GicpSolver(cfg={})", cfg_.Repr());
}

GicpStatus GicpSolver::Register(SweepGrid& grid,
                                Trajectory& traj,
                                const ImuQueue& imuq,
                                const PanoWindow& pwin,
                                const Projection& proj,
                                int gsize) {
  CHECK(!traj.empty());
  CHECK(traj.ok());
  CHECK(!proj.empty());
  CHECK(!pwin.empty());
  CHECK_EQ(grid.rows(), matches_.rows());
  CHECK_EQ(grid.cols(), matches_.cols());

  gsize_ = gsize;

  GicpStatus status;

  // how many panos to use
  const int npanos = cfg_.use_all_panos ? pwin.size() : 1;

  for (int outer = 0; outer < cfg_.max_outer_iters; ++outer) {
    // First interplate grid tfs using the updated traj
    // Note we interpolate at center point
    traj.Interp2(grid.tfs(), grid.span().end, 0.5);

    SE3d dtf_l{};  // initialize to identity, update during inner loop

    // pinds has the same shape as matches. It indicates which pano in the
    // window the current match is associated to. For example, if a point in the
    // sweep is already matched to pano 0 in the window, then we will not try to
    // match it to pano 1. On the other hand, if it is not matched to pano 0,
    // then we will try to match it to pano 1 if possible.

    for (int inner = 0; inner < cfg_.max_inner_iters; ++inner) {
      pinds_.setTo(kBadPind);

      const auto hess = BuildRigid(grid, pwin, proj, dtf_l);

      status.num_panos = npanos;
      status.num_costs = hess.n;
      status.cost = hess.c;

      if (hess.n < 20) {
        status.ok = false;
        status.msg = "Too few costs";
        return status;
      }

      // Now we solve for delta (x = [er, et])
      const auto dx = hess.Solve();
      VLOG(15) << fmt::format("x: {}", dx.transpose());
      dtf_l = dtf_l * SE3d{SO3d::exp(dx.head<3>()), dx.tail<3>()};

      ++status.num_iters;
      VLOG(12) << fmt::format("[O {} I {}] {}", outer, inner, status.Repr());

      // Check for early stop
      const int scale = cfg_.max_outer_iters - outer;
      if (CheckEarlyStop(dx, scale)) {
        VLOG(2) << fmt::format(LogColor::kBrightBlue,
                               "[Register] Early stop at O {} I {}",
                               outer,
                               inner);
        break;
      }
    }  // inner

    // Update traj with dtf and repredict
    traj.UpdateFirst(dtf_l);
    traj.PredictFull(imuq);
  }  // outer

  status.ok = true;
  status.msg = "ICP ok";
  return status;
}

Hess1 GicpSolver::BuildRigid(const SweepGrid& grid,
                             const PanoWindow& pwin,
                             const Projection& proj,
                             const Sophus::SE3d& dtf) {
  Hess1 hess;

  // map ind to pind, favor newer panos if they have enough sweeps
  auto ind2pind = [&pwin](int i) {
    const auto num = pwin.size();
    const auto& last = pwin.last();
    if (num <= 1 || last.num_sweeps() < 20) return i;
    return num - i - 1;
  };

  tfs_p_l_all_.resize(pwin.size());

  for (int i = 0; i < pwin.size(); ++i) {
    const auto pind = ind2pind(i);
    const auto& pano = pwin.At(pind);
    CHECK_GE(pano.id(), 0);  // make sure this is a valid pano

    // Skip pano that does not have enough sweeps
    if (pano.num_sweeps() < 1) continue;

    hess += BuildRigid(grid, pano, proj, dtf, pind);
  }  // pano

  return hess;
}

Hess1 GicpSolver::BuildRigid(const SweepGrid& grid,
                             const DepthPano& pano,
                             const Projection& proj,
                             const Sophus::SE3d& dtf,
                             int pind) {
  // We can compute and save the latest tfs from lidar to this pano
  // CacheTransforms(grid.tfs(), pano.tf_o_p(), dtf);
  auto& tfs_p_l = tfs_p_l_all_.at(pind);
  CacheTransforms(grid.tfs(), pano.tf_o_p(), dtf, tfs_p_l);

  const auto half_px = cfg_.half_px();
  const auto win_size = cfg_.win_size2d();
  const auto min_pano_points = cfg_.min_area();

  VLOG(13) << fmt::format(
      "[BuildRigid] half_px=({},{}), win_size=({},{}), min_area={}",
      half_px.x,
      half_px.y,
      win_size.width,
      win_size.height,
      min_pano_points);

  auto per_row = [&](int gr, Hess1& hess) {
    for (int gc = 0; gc < grid.cols(); ++gc) {
      // Skip bad point
      const auto& point = grid.PointAt(gr, gc);
      if (point.bad()) continue;

      // Skip if this point is already matched to a previous pano
      auto& match_pind = pinds_.at<uchar>(gr, gc);
      if (match_pind != kBadPind) continue;

      // x_l:  sweep point in local sweep frame
      // x_p:  pano point in pano frame
      // x_lp: sweep point in pano frame
      const auto x_l = point.mc.mean.cast<double>().eval();
      const auto& tf_p_l = tfs_p_l.at(gc);  // tf from lidar to pano
      const auto x_pl = tf_p_l * x_l;       // lidar point in pano frame
      const auto rg_pl = x_pl.norm();       // range of x_lp

      // Project to pano
      const auto px_p = ProjWithBorder(proj, x_pl, rg_pl, half_px);
      if (IsPixBad(px_p)) continue;

      auto& match = matches_.at(gr, gc);
      // This is used in match update so pull ahead
      const Matrix3d R_p_l = tf_p_l.rotationMatrix();

      // Now check if we can resue this match
      if (match.pano_id == pano.id() && match.px == px_p) {
        // If point is matched to the same pixel in the same pano, we reuse it
        // as long as it has enough matches
        CHECK_GE(match.pano_id, 0);
        CHECK(match.mc.ok());
      } else {
        // Projection ok, extract mean covar from pano
        const cv::Rect win{px_p - half_px, win_size};
        MeanCovar3f mc_p;
        const auto w = ExtractPanoWin(mc_p, proj, pano, win, rg_pl);

        // Need at least min_area points for a good match
        if (mc_p.n < min_pano_points) continue;

        match.pano_id = pano.id();
        match.weight = w;
        match.px = px_p;
        match.mc = mc_p;
      }

      // Then we can compute residual, jacobian and build hessian
      match.UpdateHess(hess, point.mc, R_p_l, x_pl);

      // Set pind to indicate this point is succesfully matched
      match_pind = static_cast<uchar>(pind);
    }
  };

  return ParallelReduce(
      {0, grid.rows(), gsize_}, Hess1{}, per_row, std::plus<>{});
}

void GicpSolver::CacheTransforms(const SE3fVec& tfs_o_l,
                                 const Sophus::SE3d& tf_o_p,
                                 const Sophus::SE3d& dtf_l,
                                 SE3dVec& tfs_p_l) {
  tfs_p_l.resize(tfs_o_l.size());
  const auto tf_p_o = tf_o_p.inverse();

  for (size_t i = 0; i < tfs_o_l.size(); ++i) {
    // T_p_l1 = T_p_o * T_o_l * T_l_l1
    const auto tf_o_l = tfs_o_l.at(i).cast<double>();
    tfs_p_l.at(i) = tf_p_o * tf_o_l * dtf_l;
  }
}

double GicpSolver::ExtractPanoWin(MeanCovar3f& mc,
                                  const Projection& proj,
                                  const DepthPano& pano,
                                  cv::Rect win,
                                  double rg_l) const {
  mc.Reset();
  // Make sure window is within bound
  win &= cv::Rect{cv::Point{}, pano.size2d()};

  const int half_rows = win.height / 2;
  const int half_cols = win.width / 2;

  int info = 0;

  for (int wr = 0; wr < win.height; ++wr) {
    for (int wc = 0; wc < win.width; ++wc) {
      const cv::Point px{wc + win.x, wr + win.y};
      const auto& data = pano.DataAt(px);
      if (data.bad()) continue;

      // Check that point in window is sufficiently close to given range
      const auto rg_p = data.GetRange();

      // Given diff range and ratio, decide which point in the window should be
      // included in computing mean and covar for matching
      const auto rg_abs_diff = std::abs(rg_p - rg_l);
      // if (rg_abs_diff > 2.0) continue;
      const auto rg_rel_diff = rg_abs_diff / rg_l;
      const auto dist = std::abs(wr - half_rows) + std::abs(wc - half_cols);
      // TODO (rofl): make these config parameters?
      if (rg_rel_diff > 0.02 * (win.width + dist)) continue;

      // Add 3d point
      const cv::Point3f pt_p = proj.Backward(px, rg_p);
      mc.Add({pt_p.x, pt_p.y, pt_p.z});
      info += data.info;
    }
  }

  // Weight is occupancy probability * range
  return static_cast<double>(info) / (pano.cfg().max_info * win.area());
}

bool GicpSolver::CheckEarlyStop(const Hess1::Vector6d& dx, int scale) const {
  const auto dr_abs_max = dx.head<3>().array().abs().maxCoeff();
  const auto dt_abs_max = dx.tail<3>().array().abs().maxCoeff();

  VLOG(13) << fmt::format("[CheckEarlyStop] dr_max: {:.3e}, dt_max: {:.3e}",
                          dr_abs_max,
                          dt_abs_max);

  return dr_abs_max < (cfg_.stop_rot_tol * scale) &&
         dt_abs_max < (cfg_.stop_pos_tol * scale);
}

void GicpSolver::Reset(cv::Range grid_span) {
  CHECK_GE(grid_span.start, 0);
  CHECK_LE(grid_span.end, matches_.cols());

  VLOG(10) << fmt::format(
      "Reset matches in grid col [{}, {})", grid_span.start, grid_span.end);

  // When we have enough new data in sweep buffer, we need to reset the
  // corresponding grid columns in matches. Because we cache match results
  // during icp so we need to clean it for new ones.

  for (int gr = 0; gr < matches_.rows(); ++gr) {
    for (int gc = grid_span.start; gc < grid_span.end; ++gc) {
      matches_.at(gr, gc).Reset();
    }
  }
}

size_t GicpSolver::Allocate(cv::Size grid_size) {
  matches_.resize(grid_size);
  pinds_.create(grid_size, CV_8UC1);
  return matches_.size() * sizeof(GicpMatch) +
         pinds_.total() * pinds_.elemSize();
}

}  // namespace sv::rofl
