#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace sv {

/// @brief Apply color map to mat
/// @details input must be 1-channel, assume after scale the max will be 1
/// default cmap is 10 = PINK. For float image it will set nan to bad_color
cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale = 1.0,
                  int cmap = cv::COLORMAP_PINK,
                  uint8_t bad_color = 255);

void ApplyCmap(const cv::Mat& mat,
               cv::Mat& disp,
               double scale = 1.0,
               int cmap = cv::COLORMAP_PINK,
               const cv::Mat& mask = {},
               uint8_t bad_color = {});

static constexpr int kImshowFlag = cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO;

/// @brief Create a window with name and show mat
void Imshow(const std::string& name,
            const cv::Mat& mat,
            int flag = kImshowFlag,
            cv::Point offset = {0, 0},
            cv::Size size = {0, 0});

/// @brief A simple keyboard control using opencv waitKey
/// @details [space] - pause/resume, [s] - step, [p] - pause
class KeyControl {
 public:
  explicit KeyControl(int wait_ms = 0,
                      const cv::Size& size = {256, 32},
                      std::string name = "control");

  /// @brief This will block if state is paused
  bool Wait();
  /// @brief A simplified version of Wait
  bool Wait2();

  int counter() const noexcept { return counter_; }

 private:
  bool paused_{true};
  int counter_{-1};
  int wait_ms_{0};
  std::string name_{"control"};

  cv::Mat disp_;
  cv::Scalar color_pause_{CV_RGB(255, 0, 0)};  // red
  cv::Scalar color_step_{CV_RGB(0, 0, 255)};   // blue
  cv::Scalar color_run_{CV_RGB(0, 255, 0)};    // green
};

/// @brief Tile cv window
class WindowTiler {
 public:
  static constexpr auto kDefaultFlags =
      cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_NORMAL | cv::WINDOW_NORMAL;

  explicit WindowTiler(const cv::Size& screen = {1920, 1200},
                       const cv::Point& offset = {0, 40})
      : screen_{screen}, offset_{offset}, curr_{offset} {}

  void Tile(const std::string& name,
            const cv::Mat& disp,
            cv::Size size = {},
            int flags = kDefaultFlags);

  void Reset() { curr_ = offset_; }

 private:
  cv::Size screen_{};
  cv::Point offset_{};
  cv::Point curr_{};
  cv::Size prev_size_{};
};

}  // namespace sv
