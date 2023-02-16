#include "sv/util/ocvviz.h"

#include "sv/util/logging.h"

namespace sv {

void Imshow(const std::string& name,
            const cv::Mat& mat,
            int flag,
            cv::Point offset,
            cv::Size size) {
  cv::namedWindow(name, flag);

  if (offset.x > 0 && offset.y > 0) {
    cv::moveWindow(name, offset.x, offset.y);
  }
  if (size.height > 0 && size.width > 0) {
    cv::resizeWindow(name, size);
  }

  cv::imshow(name, mat);
}

cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale,
                  int cmap,
                  uint8_t bad_color) {
  CHECK_EQ(input.channels(), 1);

  cv::Mat disp;
  input.convertTo(disp, CV_8UC1, scale * 255.0);
  cv::applyColorMap(disp, disp, cmap);

  if (input.depth() >= CV_32F) {
    disp.setTo(bad_color, cv::Mat(~(input > 0)));
  }

  return disp;
}

void ApplyCmap(const cv::Mat& mat,
               cv::Mat& disp,
               double scale,
               int cmap,
               const cv::Mat& bad_mask,
               uint8_t bad_color) {
  CHECK_EQ(mat.channels(), 1);
  mat.convertTo(disp, CV_8UC1, 255.0 * scale);
  cv::applyColorMap(disp, disp, cmap);  // now disp is 8UC3
  if (!bad_mask.empty()) disp.setTo(bad_color, bad_mask);
}

/// ============================================================================
namespace {

constexpr char kKeyEsc = 27;
constexpr char kKeySpace = 32;
constexpr char kKeyP = 112;
constexpr char kKeyR = 114;
constexpr char kKeyS = 115;
constexpr bool kKill = false;
constexpr bool kAlive = true;

void WriteText(cv::Mat& image,
               const std::string& text,
               cv::HersheyFonts font = cv::FONT_HERSHEY_DUPLEX) {
  cv::putText(image,
              text,
              {0, 24},                    // org
              font,                       // font
              1.0,                        // scale
              cv::Scalar(255, 255, 255),  // color
              2,                          // thick
              cv::LINE_AA);
}

}  // namespace

KeyControl::KeyControl(int wait_ms, const cv::Size& size, std::string name)
    : paused_{wait_ms > 0}, wait_ms_{wait_ms}, name_{std::move(name)} {
  disp_ = cv::Mat(size, CV_8UC3, paused_ ? color_pause_ : color_run_);
  WriteText(disp_, "press key");

  if (wait_ms_ > 0) {
    LOG(INFO)
        << "Press 's' to step, 'r' to play, 'p' to pause, 'space' to toggle "
           "play/pause, 'esc' to quit";
    cv::namedWindow(name_);
    cv::imshow(name_, disp_);
    cv::moveWindow(name_, 0, 0);
  }
}

bool KeyControl::Wait() {
  ++counter_;
  if (wait_ms_ <= 0) return true;

  const auto text = std::to_string(counter_);

  while (true) {
    if (paused_) {
      // wait forever
      auto key = cv::waitKey(0);

      switch (key) {
        case kKeySpace:
          // [space] while stop pause and return immediately
          disp_ = color_run_;
          WriteText(disp_, text);
          cv::imshow(name_, disp_);
          paused_ = false;
          return kAlive;
        case kKeyS:
          // [s] will return but keep pausing
          disp_ = color_step_;
          WriteText(disp_, text);
          cv::imshow(name_, disp_);
          paused_ = true;
          return kAlive;
        case kKeyEsc:
          return kKill;
      }
    } else {
      // not pause, so it is running
      const auto key = cv::waitKey(wait_ms_);
      // Press space or P will pause
      if (key == kKeySpace || key == kKeyP) {
        disp_ = color_pause_;
        WriteText(disp_, text);
        cv::imshow(name_, disp_);
        paused_ = true;
      } else {
        break;
      }
    }
  }

  return kAlive;
}

bool KeyControl::Wait2() {
  ++counter_;
  // If no wait then we just retun
  if (wait_ms_ <= 0) return true;

  if (!paused_) {
    auto key = cv::waitKey(wait_ms_);

    switch (key) {
      case kKeySpace:
        // Space will pause/resume
        paused_ = !paused_;
        break;
      case 'p':
        paused_ = true;
        break;
      case kKeyEsc:
      case 'q':
        // esc or q will quit
        return false;
      default:
        break;
    }
  }

  disp_ = paused_ ? color_pause_ : color_run_;
  WriteText(disp_, std::to_string(counter_));
  cv::imshow(name_, disp_);

  while (paused_) {
    auto key = cv::waitKey(0);

    switch (key) {
      case kKeySpace:
        // Space will pause/resume
        paused_ = !paused_;
        break;
      case 's':
        // s will step
        return true;
      case kKeyEsc:
      case 'q':
        // esc or q will quit
        return false;
      default:
        break;
    }
  }

  return true;
}

/// ============================================================================
void WindowTiler::Tile(const std::string& name,
                       const cv::Mat& disp,
                       cv::Size size,
                       int flags) {
  if (flags < 0) flags = kDefaultFlags;
  if (size.empty()) size = {disp.cols, disp.rows};

  if (curr_.x + size.width > screen_.width) {
    curr_.x = 0;
    curr_.y += prev_size_.height + offset_.y;
  }

  if (curr_.y + size.height > screen_.height) {
    curr_.y = 0;
  }

  cv::namedWindow(name, flags);
  cv::moveWindow(name, curr_.x, curr_.y);
  cv::resizeWindow(name, size);
  cv::imshow(name, disp);

  // update curr
  const auto rect = cv::getWindowImageRect(name);
  curr_.x += rect.width;
  prev_size_ = size;
}
}  // namespace sv
