#include "sv/rofl/odom.h"

#include "sv/util/logging.h"
#include "sv/util/memsize.h"
#include "sv/util/opencv.h"
#include "sv/util/tbb.h"

namespace sv::rofl {

namespace {

/// @brief Add all the timings in summary based on initial char
absl::Duration SumTimingsByInittial(const TimerSummary& summary, char initial) {
  absl::Duration total_time;

  for (const auto& name_stats : summary.dict()) {
    if (!name_stats.first.empty() && name_stats.first.front() == initial) {
      total_time += name_stats.second.last();
    }
  }

  return total_time;
}

}  // namespace

/// ============================================================================
const OdomCfg& OdomCfg::Check() const {
  CHECK_GE(tbb, 0);
  CHECK_GT(rate_factor, 0);
  CHECK_EQ(num_panos, 2);
  CHECK_GT(pano_min_sweeps, 0);
  CHECK_GE(pano_max_sweeps, 0);
  CHECK_GE(pano_match_ratio, 0);
  CHECK_LE(pano_match_ratio, 1);
  CHECK(IsPowerOf2(rate_factor))
      << fmt::format("Rate multiplier [{}] is not power of 2.", rate_factor);
  return *this;
}

std::string OdomCfg::Repr() const {
  return fmt::format(
      "OdomCfg(tbb={}, use_signal={}, rate_factor={}, num_panos={}, "
      "pano_min_sweeps={}, pano_max_sweeps={},pano_max_trans={}, "
      "pano_match_ratio={}, pano_align_gravity={}, pano_render_prev={})",
      tbb,
      use_signal,
      rate_factor,
      num_panos,
      pano_min_sweeps,
      pano_max_sweeps,
      pano_max_trans,
      pano_match_ratio,
      pano_align_gravity,
      pano_render_prev);
}

size_t LidarOdom::Allocate(cv::Size sweep_size) {
  // allocate sweep
  const auto sweep_bytes = sweep.Allocate(sweep_size);
  const auto grid_bytes = grid.Allocate(sweep_size);
  const auto traj_bytes = traj.Allocate(grid.cols());
  const auto gicp_bytes = gicp.Allocate(grid.size2d());
  const auto pwin_bytes = pwin.Allocate(cfg_.num_panos, proj.size2d());

  total_bytes_ =
      sweep_bytes + grid_bytes + traj_bytes + gicp_bytes + pwin_bytes;

  LOG(INFO) << fmt::format(LogColor::kBrightRed,
                           "Memory (MB): sweep {}, grid {}, traj {}, gicp {}, "
                           "pwin {}, total {}",
                           Bytes(sweep_bytes),
                           Bytes(grid_bytes),
                           Bytes(traj_bytes),
                           Bytes(gicp_bytes),
                           Bytes(pwin_bytes),
                           Bytes(total_bytes_));

  return total_bytes_;
}

void LidarOdom::AddScan(const LidarScan& scan) {
  LOG_IF(WARNING, scan.cols() > max_cols())
      << fmt::format("Scan cols {} > max_cols {}", scan.cols(), max_cols());

  // Detect jump
  if (scan.span().start != (buf_span_.end % sweep.cols())) {
    LOG(WARNING) << fmt::format(LogColor::kBrightRed,
                                "Detect jump {} -> {}",
                                buf_span_.end,
                                scan.span().start);

    // TODO (rofl): Handle jump by clear bad area
  }

  // Add scan to sweep
  {
    auto _ = ts_.Scoped("A0_AddScan");
    sweep.Add(scan);
  }

  // Fill small holes in range and signal
  //  int n_fill{};
  //  if (cfg_.fill_holes) {
  //    auto _ = ts.Scoped("A1_FillHoles");
  //    n_fill = sweep.FillHoles(gsize);
  //  }

  // Compute range 2nd gradients
  {
    auto _ = ts_.Scoped("A2_RangeGrad");
    sweep.CalcRangeGrad2(cfg_.tbb);
  }

  // Compute signal gradient
  if (cfg_.use_signal) {
    auto _ = ts_.Scoped("A2_SignalGrad");
    sweep.CalcSignalGrad(cfg_.tbb);
    // Compute signal edge
    // sweep.CalcSignalEdge(50, cfg_.tbb);
  }

  // Selected points in the new part of the sweep buffer
  // note this is different from info_.num_selected, which is the total number
  // of selected points within the sweep buffer
  int n_select{};
  {
    auto t = ts_.Scoped("A3_SelectPixels");
    n_select = grid.Select(sweep, cfg_.tbb);
  }

  // Update buffer span with newly added scan. Once UpdateMap() is called, the
  // part which will be overwritten by the next incoming scans, will be added to
  // the panos.
  buf_span_.end += scan.cols();
  CHECK_LE(buf_span_.end, sweep.cols());

  VLOG(1) << fmt::format("[AddScan] scan {}x{}={}, select {}, span [{},{})={}",
                         scan.rows(),
                         scan.cols(),
                         scan.total(),
                         n_select,
                         buf_span_.start,
                         buf_span_.end,
                         buf_span_.size());
}

bool LidarOdom::Estimate() {
  VLOG(1) << fmt::format("[Estimate] Span full [{},{}) >= max_cols {}",
                         buf_span_.start,
                         buf_span_.end,
                         max_cols());

  // Number of new states to predict
  const auto buf_cols = buf_span_.size();               // new cols in buffer
  const auto t0 = sweep.TimeFromEnd(buf_cols);          // start time
  const auto t1 = sweep.TimeFromEnd(0);                 // end time
  const int seg_cols = sweep.cols() / traj.segments();  // cols per segment
  const int num_segs = buf_cols / seg_cols;             // num segs to predict

  IntPair imu_ind_range{};  // start and end index of imus used for prediction
  {
    auto _ = ts_.Scoped("E0_Predict");
    imu_ind_range = traj.PredictNew(imuq, {t0, t1}, num_segs);
  }
  VLOG(1) << fmt::format(
      "[Predict] segments: {}, time: [{:.4f}, {:.4f}], delta_time: {:.4f}, "
      "imus: [{},{})/{}",
      num_segs,
      t0,
      t1,
      t1 - t0,
      imu_ind_range.first,
      imu_ind_range.second,
      imuq.size());

  if (!traj.ok()) {
    LOG(WARNING) << fmt::format(LogColor::kYellow,
                                "[Estimate] traj is not ok, skip");
    return false;
  }

  if (pwin.empty()) {
    LOG(WARNING) << fmt::format(LogColor::kYellow,
                                "[Estimate] pwin is empty, skip");
    return false;
  }

  GicpStatus status;
  {
    auto _ = ts_.Scoped("E1_Register");
    // We need to reset part of the matches that corresponds to the current
    // buffer span. This way, if we are using partial sweeps, we can resue
    // previously matched results
    gicp.Reset(buf_span_ / grid.cfg().cell_cols);
    status = gicp.Register(grid, traj, imuq, pwin, proj, cfg_.tbb);
  }
  VLOG(1) << "[Estimate] " << status.Repr();

  // Update odom info
  info_.num_selected = grid.GetNumSelected();
  info_.num_matched = status.num_costs;
  info_.add_pano = false;
  return true;
}

bool LidarOdom::ShouldAddPano(const Sophus::SE3d& tf_o_l) {
  msg_add_pano_.clear();

  // Add one if last pano is empty
  if (pwin.empty()) {
    msg_add_pano_ = "pano window is empty";
    return true;
  }

  // Now we must have at least one pano, we check num_sweeps
  const auto last_num_sweeps = pwin.last().num_sweeps();
  if (cfg_.pano_min_sweeps > 0 && last_num_sweeps < cfg_.pano_min_sweeps) {
    msg_add_pano_ = fmt::format(
        "last sweeps {} < {}", last_num_sweeps, cfg_.pano_min_sweeps);
    return false;
  }

  const auto first_num_sweeps = pwin.first().num_sweeps();
  if (cfg_.pano_max_sweeps > 0 && first_num_sweeps > cfg_.pano_max_sweeps) {
    msg_add_pano_ = fmt::format(
        "first sweeps {} > {}", first_num_sweeps, cfg_.pano_max_sweeps);
    return true;
  }

  // Then we check match ratio
  const auto match_ratio = info_.match_ratio();
  CHECK_GT(match_ratio, 0) << info_.Repr();
  CHECK_LE(match_ratio, 1) << info_.Repr();

  VLOG(2) << fmt::format("[AddPano] select: {}, match: {}, ratio: {:.2f}",
                         info_.num_selected,
                         info_.num_matched,
                         match_ratio);

  if (match_ratio < cfg_.pano_match_ratio) {
    msg_add_pano_ = fmt::format(
        "match_ratio {:.3f} < {}", match_ratio, cfg_.pano_match_ratio);
    return true;
  }

  // Finally we check if translation from last pano is too big
  const auto& tf_o_p = pwin.last().tf_o_p();
  const auto tf_p_l = tf_o_p.inverse() * tf_o_l;
  const auto trans = tf_p_l.translation().norm();
  if (trans > cfg_.pano_max_trans) {
    msg_add_pano_ =
        fmt::format("trans {:.3f} > {}", trans, cfg_.pano_max_trans);
    return true;
  }

  return false;
}

void LidarOdom::UpdateMap() {
  // This is the part that will be ejected and added to panos
  cv::Range add_span;
  add_span.start = buf_span_.end % sweep.cols();
  add_span.end = add_span.start + buf_span_.size();

  // Undistort the entire sweep (for visualization)
  {
    auto _ = ts_.Scoped("U0_Undistort");
    traj.Interp2(sweep.tfs(), sweep.span().end, 0, cfg_.tbb);
  }

  auto tf_o_l = traj.GetTfOdomLidar();
  if (ShouldAddPano(tf_o_l)) {
    // Remove the oldest one if window is full
    if (pwin.full()) {
      const auto& pano_rm = pwin.RemoveFront();
      msg_rm_pano_ = fmt::format(
          "remove {}, sweeps {}", pano_rm.id(), pano_rm.num_sweeps());
    }

    // Add pano
    if (cfg_.pano_align_gravity) {
      tf_o_l.so3() = traj.GetRotWorldOdom().inverse();
    }
    auto& pano_new = pwin.AddPano(pano_id_++, sweep.TimeEndNs(), tf_o_l);

    // Render previous pano into new one
    if (cfg_.pano_render_prev && pwin.size() >= 2) {
      // Only render a valid removed pano
      int n_render = 0;
      {
        auto _ = ts_.Scoped("R0_PanoRender");
        const auto& pano_prev = pwin.At(pwin.size() - 2);
        n_render = pano_new.RenderPano(pano_prev, proj, cfg_.tbb);
      }
      LOG(INFO) << fmt::format("[UpdateMap] Render {} points to new pano",
                               n_render);
    }

    LOG(INFO) << fmt::format("[UpdateMap] add {}, {}, {}",
                             pano_new.id(),
                             msg_rm_pano_,
                             msg_add_pano_);
    info_.add_pano = true;
  }

  // At this point we must have at least one pano
  CHECK(!pwin.empty());

  // Add ejected partial sweep to both pano
  {
    auto _ = ts_.Scoped("U1_PanoAdd");
    ParallelFor({0, pwin.size(), cfg_.tbb}, [&](int i) {
      pwin.At(i).AddSweep(sweep, proj, add_span, cfg_.tbb);
    });
  }

  //  const auto n_points = add_span.size() * sweep.rows();
  //  VLOG(1) << fmt::format(
  //      "[UpdateMap] add {} out of {} points from buf [{}, {}) to pano "
  //      "({:.2f}%), pano sweeps: {}",
  //      n_add,
  //      n_points,
  //      add_span.start,
  //      add_span.end,
  //      (n_add * 100.0) / n_points,
  //      pano.num_sweeps());

  // reset buffer span
  buf_span_.start = buf_span_.end % sweep.cols();
  buf_span_.end = buf_span_.start;

  // Collect last AddScan and Estimate to be odom and
  LogOdomAndMapTimes();
}

std::string LidarOdom::Timings() const { return ts_.ReportAll(true); }

void LidarOdom::LogOdomAndMapTimes() {
  const auto t_odom = SumTimingsByInittial(ts_, 'E');
  const auto t_map = SumTimingsByInittial(ts_, 'U');
  ts_.Add("_Odometry", t_odom);
  ts_.Add("_Mapping", t_map);
}

std::string OdomInfo::Repr() const {
  return fmt::format("OdomInfo(num_selected={}, num_matched={}, add_pano={})",
                     num_selected,
                     num_matched,
                     add_pano);
}

}  // namespace sv::rofl

