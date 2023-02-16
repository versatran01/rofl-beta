#pragma once

#include <absl/container/inlined_vector.h>

#include "sv/rofl/pano.h"

namespace sv::rofl {

class PanoWindow {
 private:
  int p_{};  // points to one past last pano
  absl::InlinedVector<DepthPano*, 4> ptrs_;
  std::vector<DepthPano> panos_;

 public:
  PanoWindow() = default;
  explicit PanoWindow(int num_panos) { Resize(num_panos); }
  PanoWindow(int num_panos, cv::Size pano_size) {
    Allocate(num_panos, pano_size);
  }

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const PanoWindow& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Info
  int size() const noexcept { return p_; }
  bool empty() const noexcept { return p_ == 0; }
  bool full() const noexcept { return p_ >= capacity(); }
  int capacity() const noexcept { return static_cast<int>(panos_.size()) - 1; }

  /// @brief Getters
  DepthPano& At(int i);
  const DepthPano& At(int i) const;

  const DepthPano& first() const { return At(0); }
  const DepthPano& last() const { return At(p_ - 1); }
  const DepthPano& removed() const { return *ptrs_.back(); }

  /// @brief Add a new pano
  DepthPano& AddPano(int id, double time, const Sophus::SE3d& tf_o_p);
  /// @brief Remove pano at index
  /// @note This will not erase the pano, only move it to the remove slot
  DepthPano& RemovePanoAt(int i);
  DepthPano& RemoveFront() { return RemovePanoAt(0); }

  /// @brief Clear window
  void Reset() noexcept { p_ = 0; }
  /// @brief Resize window, does not allocate
  void Resize(int num_panos);
  /// @brief Resize window and allocate keyframes
  size_t Allocate(int num_panos, cv::Size pano_size);
};

}  // namespace sv::rofl
