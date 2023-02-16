#pragma once

#include "sv/rofl/gicp.h"
#include "sv/rofl/grid.h"
#include "sv/rofl/pwin.h"
#include "sv/rofl/scan.h"
#include "sv/util/ocvviz.h"

namespace sv::rofl {

struct IvizCfg {
  double min_range{1.0};
  double max_signal{512.0};
  double disp_scale_sweep{1.0};
  double disp_scale_pano{1.0};
  int screen_width{1920};
  int screen_height{1280};
  bool show_pano_range{true};
  bool show_pano_signal{true};
  bool show_pano_info{false};
  bool show_pano_grad{false};

  const IvizCfg& Check() const;
  std::string Repr() const;
};

struct PanoViz {
  cv::Mat range;
  cv::Mat signal;
  cv::Mat info;
  cv::Mat grad;
};

class Visualizer {
 private:
  IvizCfg cfg_;
  WindowTiler tiler_;
  bool show_sweep_{false};
  bool show_pano_{false};

  // sweep
  cv::Mat sweep_range_;
  cv::Mat sweep_range_disp_;
  cv::Mat sweep_range_gx2_disp_;
  cv::Mat sweep_signal_;
  cv::Mat sweep_signal_disp_;
  cv::Mat sweep_edge_disp_;
  cv::Mat sweep_select_mask_;

  // pano
  std::vector<PanoViz> pviz_;

 public:
  explicit Visualizer(const IvizCfg& cfg = {})
      : cfg_(cfg.Check()), tiler_({cfg.screen_width, cfg.screen_height}) {}

  const auto& cfg() const noexcept { return cfg_; }

  /// @group Draw
  void Reset() { tiler_.Reset(); }
  void DrawSweep(const LidarSweep& sweep, const SweepGrid& grid);
  void DrawPanos(const PanoWindow& pwin, const GicpSolver& gicp);
  void Display();

 private:
  void DrawPano(const DepthPano& pano, const GicpSolver& gicp, PanoViz& pviz);
  void DrawMatches(const MatchGrid& matches, int pano_id, PanoViz& pviz);
};

/// @brief Convert range image (16UC1) to color with log scale
void ColorRange16ULog(const cv::Mat& r16u,
                      cv::Mat& disp,
                      double range_scale,
                      double min_range,
                      double max_range);

/// @brief Convert range image (16UC1) to color with inverse range
void ColorRange16UInv(const cv::Mat& r16u, cv::Mat& disp, double range_scale);

void DrawVerticalLine(cv::Mat& disp,
                      int x,
                      cv::Range y_range,
                      const cv::Scalar& color,
                      int thickness = 1);

}  // namespace sv::rofl
