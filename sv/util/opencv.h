#pragma once

#include <opencv2/core/mat.hpp>

namespace sv {

/// @brief Get the corresponding cv enum value given type
/// @example cv_type<cv::Vec3f>::value == CV_32FC3
///          cv_type_v<cv::Vec3f> == CV_32FC3
template <typename T>
struct cv_type;

template <>
struct cv_type<uchar> {
  static constexpr int value = CV_8U;
};

template <>
struct cv_type<schar> {
  static constexpr int value = CV_8S;
};

template <>
struct cv_type<ushort> {
  static constexpr int value = CV_16U;
};

template <>
struct cv_type<int16_t> {
  static constexpr int value = CV_16S;
};

template <>
struct cv_type<int> {
  static constexpr int value = CV_32S;
};

template <>
struct cv_type<float> {
  static constexpr int value = CV_32F;
};

template <>
struct cv_type<double> {
  static constexpr int value = CV_64F;
};

template <typename T, int N>
struct cv_type<cv::Vec<T, N>> {
  static constexpr int value = (CV_MAKETYPE(cv_type<T>::value, N));
};

template <typename T>
inline constexpr int cv_type_v = cv_type<T>::value;

/// @brief Convert cv::Mat::type() to string
/// @example CvTypeStr(CV_8UC1) == "8UC1"
std::string CvTypeStr(int type) noexcept;

/// @brief Get mat size
inline cv::Size CvMatSize(const cv::Mat& mat) noexcept {
  return {mat.cols, mat.rows};
}

/// Range * d
inline cv::Range& operator*=(cv::Range& lhs, int d) noexcept {
  lhs.start *= d;
  lhs.end *= d;
  return lhs;
}
inline cv::Range operator*(cv::Range lhs, int d) noexcept { return lhs *= d; }

/// Range / d
inline cv::Range& operator/=(cv::Range& lhs, int d) noexcept {
  lhs.start /= d;
  lhs.end /= d;
  return lhs;
}
inline cv::Range operator/(cv::Range lhs, int d) noexcept { return lhs /= d; }

/// Size / Size
inline cv::Size& operator/=(cv::Size& lhs, const cv::Size& rhs) noexcept {
  lhs.width /= rhs.width;
  lhs.height /= rhs.height;
  return lhs;
}

inline cv::Size operator/(cv::Size lhs, const cv::Size& rhs) noexcept {
  return lhs /= rhs;
}

inline cv::Mat CvZeroLike(const cv::Mat& mat) {
  return cv::Mat::zeros(mat.rows, mat.cols, mat.type());
}

/// @brief Set roi in mat to val
template <typename T>
bool MatSetRoi(cv::Mat& mat, cv::Rect roi, const T& val) noexcept {
  roi &= cv::Rect{0, 0, mat.cols, mat.rows};
  if (roi.empty()) return false;

  for (int r = 0; r < roi.height; ++r) {
    for (int c = 0; c < roi.width; ++c) {
      mat.at<T>(roi.y + r, roi.x + c) = val;
    }
  }
  return true;
}

/// @brief Set window in image to val, given center and half size
template <typename T>
bool MatSetWin(cv::Mat& mat,
               const cv::Point& px,
               const cv::Point& half,
               const T& val) noexcept {
  cv::Size size{half.x * 2 + 1, half.y * 2 + 1};
  cv::Rect roi{px - half, size};
  return MatSetRoi(mat, roi, val);
}

template <typename T>
double PointSqNorm(const cv::Point_<T>& p) noexcept {
  return p.x * p.x + p.y * p.y;
}

template <typename T>
double PointSqNorm(const cv::Point3_<T>& p) noexcept {
  return p.x * p.x + p.y * p.y + p.z * p.z;
}

}  // namespace sv
