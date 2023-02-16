#include "sv/rofl/iviz.h"

#include "sv/util/logging.h"

namespace sv::rofl {

namespace {

cv::Size GetScaledSize(const cv::Mat& mat, double scale) {
  return {static_cast<int>(mat.cols * scale),
          static_cast<int>(mat.rows * scale)};
}

}  // namespace

const IvizCfg& IvizCfg::Check() const {
  CHECK_GT(min_range, 0);
  CHECK_GT(disp_scale_pano, 0);
  CHECK_GT(disp_scale_sweep, 0);
  CHECK_GT(screen_width, 0);
  CHECK_GT(screen_height, 0);
  return *this;
}

std::string IvizCfg::Repr() const {
  return fmt::format(
      "IvizCfg(min_range={}, max_signal={}, "
      "disp_scale_sweep={}, disp_scale_pano={}, "
      "show_pano_signal={}, show_pano_info={}, "
      "screen_width={}, screen_height={})",
      min_range,
      max_signal,
      disp_scale_sweep,
      disp_scale_pano,
      show_pano_signal,
      show_pano_info,
      screen_width,
      screen_height);
}

void DrawVerticalLine(cv::Mat& disp,
                      int x,
                      cv::Range y_range,
                      const cv::Scalar& color,
                      int thickness) {
  CHECK_EQ(disp.type(), CV_8UC3);
  cv::line(
      disp, {x, y_range.start}, {x, y_range.end}, color, thickness, cv::LINE_4);
}

void ColorRange16ULog(const cv::Mat& r16u,
                      cv::Mat& disp,
                      double range_scale,
                      double min_range,
                      double max_range) {
  CHECK_EQ(r16u.type(), CV_16UC1);
  r16u.convertTo(disp, CV_32FC1, 1.0 / range_scale, 1);
  cv::log(disp, disp);
  const double rmax = std::log1p(max_range);
  const double rmin = std::log1p(min_range);
  disp = (disp - rmin) / (rmax - rmin);
  ApplyCmap(disp, disp, 1.0, cv::COLORMAP_PINK);
}

void ColorRange16UInv(const cv::Mat& r16u, cv::Mat& disp, double range_scale) {
  CHECK_EQ(r16u.type(), CV_16UC1);
  r16u.convertTo(disp, CV_32FC1, 1.0 / range_scale, 1);
  disp = 1.0 / disp;
  ApplyCmap(disp, disp, 1.0, cv::COLORMAP_PINK);
}

void Visualizer::DrawSweep(const LidarSweep& sweep, const SweepGrid& grid) {
  show_sweep_ = true;

  const auto signal_scale = 1.0 / cfg_.max_signal;
  const auto range_scale = sweep.info().range_scale * cfg_.min_range;

  // Extract range part and convert to color
  sweep.ExtractRange16u(sweep_range_);
  ColorRange16UInv(sweep_range_, sweep_range_disp_, range_scale);

  //  If use partial sweep, we mark the start and end of the col rnage
  if (sweep.IsPartial()) {
    // Green marks start
    DrawVerticalLine(sweep_range_disp_,
                     sweep.span().start,
                     {0, sweep.cols()},
                     CV_RGB(0, 255, 0));
    // Red marks end
    DrawVerticalLine(sweep_range_disp_,
                     sweep.span().end - 1,
                     {0, sweep.cols()},
                     CV_RGB(255, 0, 0));
  }

  grid.MakeSelectMask(sweep_select_mask_);

  // Extract signal part and convert to color
  sweep.ExtractSignal16u(sweep_signal_);
  ApplyCmap(sweep_signal_,
            sweep_signal_disp_,
            signal_scale,
            cv::COLORMAP_AUTUMN,
            sweep_signal_ == 0,
            255);

  // Visualize range 2nd order gradient (abs value)
  ApplyCmap(sweep.ddrdu(),
            sweep_range_gx2_disp_,
            10.0,  // scale
            cv::COLORMAP_AUTUMN,
            ~(sweep.ddrdu() < 128.0),
            255);

  // Visualize signal edge
  // ApplyCmap(sweep.edge() + 6, sweep_edge_disp_, 1 / 12.0,
  // cv::COLORMAP_VIRIDIS);
}

void Visualizer::DrawPanos(const PanoWindow& pwin, const GicpSolver& gicp) {
  pviz_.resize(pwin.size());
  for (int i = 0; i < pwin.size(); ++i) {
    DrawPano(pwin.At(i), gicp, pviz_.at(i));
  }
}

void Visualizer::DrawPano(const DepthPano& pano,
                          const GicpSolver& gicp,
                          PanoViz& pviz) {
  if (pano.num_sweeps() == 0) return;
  show_pano_ = true;

  const auto& pano_cfg = pano.cfg();

  if (cfg_.show_pano_range) {
    pano.ExtractRange16U(pviz.range);
    ColorRange16UInv(pviz.range, pviz.range, PanoData::kRangeScale);
  }

  if (cfg_.show_pano_signal) {
    pano.ExtractSignal16U(pviz.signal);
    ApplyCmap(
        pviz.signal, pviz.signal, 1.0 / cfg_.max_signal, cv::COLORMAP_VIRIDIS);
  }

  if (cfg_.show_pano_info) {
    pano.ExtractCount16U(pviz.info);
    ApplyCmap(pviz.info,
              pviz.info,
              1.0 / (pano_cfg.max_info + 1.0),
              cv::COLORMAP_OCEAN);
  }

  if (cfg_.show_pano_grad) {
    pano.ExtractGrad16U(pviz.grad);
    ApplyCmap(pviz.grad, pviz.grad, 1.0 / 256, cv::COLORMAP_CIVIDIS);
  }

  // Draw match
  DrawMatches(gicp.matches(), pano.id(), pviz);
}

void Visualizer::DrawMatches(const MatchGrid& matches,
                             int pano_id,
                             PanoViz& pviz) {
  for (const auto& match : matches) {
    // Make sure id matches
    if (match.pano_id != pano_id) continue;

    const cv::Rect rect{match.px.x - 1, match.px.y - 1, 3, 3};
    const auto color = match.mc.ok() ? CV_RGB(0, 255, 0) : CV_RGB(255, 0, 0);
    cv::rectangle(pviz.range, rect, color, 1, cv::LINE_4);
  }
}

void Visualizer::Display() {
  tiler_.Reset();

  if (show_sweep_) {
    const auto disp_size = GetScaledSize(sweep_range_, cfg_.disp_scale_sweep);
    tiler_.Tile("sweep_range", sweep_range_disp_, disp_size);
    tiler_.Tile("sweep_flat", sweep_select_mask_, disp_size);
    tiler_.Tile("sweep_ddr", sweep_range_gx2_disp_, disp_size);
    tiler_.Tile("sweep_signal", sweep_signal_disp_, disp_size);
  }

  if (show_pano_) {
    for (size_t i = 0; i < pviz_.size(); ++i) {
      const auto& pviz = pviz_.at(i);
      const auto disp_size = GetScaledSize(pviz.range, cfg_.disp_scale_pano);
      const auto prefix = fmt::format("pano{}_", i);

      if (!pviz.range.empty()) {
        tiler_.Tile(prefix + "range", pviz.range, disp_size);
      }

      if (!pviz.signal.empty()) {
        tiler_.Tile(prefix + "signal", pviz.signal, disp_size);
      }

      if (!pviz.info.empty()) {
        tiler_.Tile(prefix + "info", pviz.info, disp_size);
      }

      if (!pviz.grad.empty()) {
        tiler_.Tile(prefix + "grad", pviz.grad, disp_size);
      }
    }
  }

  cv::waitKey(1);
}

}  // namespace sv::rofl
