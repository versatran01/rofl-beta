#pragma once

#include <opencv2/core/types.hpp>

namespace sv {

struct Grid2dBase {
  using Size2d = cv::Size;
  using Point2 = cv::Point;
  static_assert(sizeof(Size2d) == 8);
  static_assert(sizeof(Point2) == 8);

 protected:
  Size2d size2d_{};

 public:
  Grid2dBase() = default;
  Grid2dBase(int rows, int cols) : size2d_{cols, rows} {}
  explicit Grid2dBase(Size2d size2d) : size2d_{size2d} {}

  auto size2d() const noexcept { return size2d_; }
  auto area() const noexcept { return size2d_.area(); }
  bool empty() const noexcept { return area() == 0; }
  auto size() const noexcept { return static_cast<size_t>(area()); }

  int cols() const noexcept { return size2d_.width; }
  int rows() const noexcept { return size2d_.height; }
  int width() const noexcept { return size2d_.width; }
  int height() const noexcept { return size2d_.height; }

  int rc2ind(int r, int c) const noexcept { return r * cols() + c; }
};

template <typename T>
class Grid2d final : public Grid2dBase {
  std::vector<T> data_{};

 public:
  using container = std::vector<T>;
  using value_type = typename container::value_type;
  using pointer = typename container::pointer;
  using const_pointer = typename container::const_pointer;
  using reference = typename container::reference;
  using const_reference = typename container::const_reference;
  using iterator = typename container::iterator;
  using const_iterator = typename container::const_iterator;
  using const_reverse_iterator = typename container::const_reverse_iterator;
  using reverse_iterator = typename container::reverse_iterator;
  using size_type = typename container::size_type;
  using difference_type = typename container::difference_type;
  using allocator_type = typename container::allocator_type;

  Grid2d() = default;
  Grid2d(int rows, int cols, const T& val = {})
      : Grid2dBase{rows, cols}, data_(rows * cols, val) {}
  explicit Grid2d(Size2d size2d, const T& val = {})
      : Grid2d{size2d.height, size2d.width, val} {}

  void reset(const T& val = {}) { data_.assign(size(), val); }

  void reserve(Size2d size2d) const { data_.reserve(size2d.area()); }

  void resize(Size2d size2d, const T& val = {}) {
    size2d_ = size2d;
    data_.resize(area(), val);
  }

  size_type bytes() const noexcept { return area() * sizeof(T); }

  reference at(size_t i) { return data_.at(i); }
  const_reference at(size_t i) const { return data_.at(i); }

  reference at(Point2 pt) { return at(pt.y, pt.x); }
  const_reference at(Point2 pt) const { return at(pt.y, pt.x); }

  reference at(int r, int c) { return at(rc2ind(r, c)); }
  const_reference at(int r, int c) const { return at(rc2ind(r, c)); }

  pointer data() noexcept { return data_.data(); };
  const_pointer data() const noexcept { return data_.data(); }

  const_reference front() const { return at(0); }
  const_reference back() const { return at(size() - 1); }

  iterator begin() noexcept { return data_.begin(); }
  iterator end() noexcept { return data_.end(); }

  const_iterator begin() const noexcept { return data_.begin(); }
  const_iterator end() const noexcept { return data_.end(); }

  const_iterator cbegin() const { return data_.cbegin(); }
  const_iterator cend() const noexcept { return data_.cend(); }
};

}  // namespace sv
