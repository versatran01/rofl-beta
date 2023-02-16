#pragma once

#include <cstdint>
#include <iostream>
#include <string>

namespace sv {

/// @brief Format bytes to human readable representation
std::string FormatBytes(int64_t bytes);

/// @class Similar to absl duration except for memmory size
class Memsize {
  using size_type = int64_t;
  size_type bytes_{};

 public:
  static constexpr int64_t kKilo = 1024;
  static constexpr int64_t kMega = 1024 * 1024;
  static constexpr int64_t kGiga = 1024 * 1024 * 1024;

  constexpr Memsize(size_type bytes = 0) : bytes_(bytes) {}
  size_type bytes() const noexcept { return bytes_; }
  std::string Repr() const { return FormatBytes(bytes_); }

  friend std::ostream& operator<<(std::ostream& os, Memsize m) {
    return os << m.Repr();
  }
};

/// @group
/// Bytes()
/// KiloBytes()
/// MegaBytes()
/// GigaBytes()
///
/// Factory functions for constructing `Memsize` values from an integral number
/// of the unit indicated by the factory function's name. The number must be
/// representable as int64_t.
constexpr Memsize Bytes(int64_t n) { return {n}; }
constexpr Memsize KiloBytes(int64_t n) { return {n * Memsize::kKilo}; }
constexpr Memsize MegaBytes(int64_t n) { return {n * Memsize::kMega}; }
constexpr Memsize GigaBytes(int64_t n) { return {n * Memsize::kGiga}; }

}  // namespace sv
