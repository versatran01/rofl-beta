#include "sv/rofl/pwin.h"

#include "sv/util/logging.h"

namespace sv::rofl {

using SE3d = Sophus::SE3d;

std::string PanoWindow::Repr() const {
  return fmt::format("PanoWindow(size={}/{})", size(), capacity());
}

DepthPano& PanoWindow::At(int i) {
  CHECK_LE(0, i);
  CHECK_LT(i, capacity());
  return *ptrs_.at(i);
}

const DepthPano& PanoWindow::At(int i) const {
  CHECK_LE(0, i);
  CHECK_LT(i, capacity());
  return *ptrs_.at(i);
}

DepthPano& PanoWindow::AddPano(int id, double time, const SE3d& tf_o_p) {
  CHECK(!full());
  CHECK_LE(0, p_);
  CHECK_GT(time, 0);

  // note that p_ is 1 past current pano
  auto& pano = At(p_);
  pano.Reset(id, tf_o_p);
  ++p_;
  return pano;
}

DepthPano& PanoWindow::RemovePanoAt(int i) {
  CHECK_LE(0, i);
  CHECK_LT(i, size());

  // rotate left by 1 from this pano to last
  // Imagine we have 3 valid panos [0, 1, 2] and we remove 1 (p points at 3)
  // Max pano size is 4
  //     x     p
  // [0, 1, 2, 3, 4]
  // after rotate we have [0, 2] and 1 should be at the end
  //        p     x
  // [0, 2, 3, 4, 1]
  // std::rotate(first, new_first, last)
  std::rotate(ptrs_.begin() + i, ptrs_.begin() + i + 1, ptrs_.end());
  --p_;
  return *ptrs_.back();
}

void PanoWindow::Resize(int num_panos) {
  // actually allocate num_panos + 1 kf to store the removed pano
  const auto size = num_panos + 1;
  ptrs_.resize(size);
  panos_.resize(size);
  for (int i = 0; i < size; ++i) {
    ptrs_.at(i) = panos_.data() + i;
  }
}

size_t PanoWindow::Allocate(int num_panos, cv::Size pano_size) {
  Resize(num_panos);

  size_t bytes = 0;
  for (auto& pano : panos_) {
    bytes += pano.Allocate(pano_size);
  }
  return bytes;
}

}  // namespace sv::rofl
