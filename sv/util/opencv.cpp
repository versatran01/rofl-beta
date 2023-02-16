#include "sv/util/opencv.h"

#include "sv/util/logging.h"

namespace sv {

std::string CvTypeStr(int type) noexcept {
  cv::Mat a;
  std::string r;

  const uchar depth = type & CV_MAT_DEPTH_MASK;
  const auto chans = static_cast<uchar>(1 + (type >> CV_CN_SHIFT));

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  return fmt::format("{}C{}", r, chans);
}

}  // namespace sv
