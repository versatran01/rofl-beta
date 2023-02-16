#include "sv/util/memsize.h"

#include <fmt/core.h>

namespace sv {

namespace {
constexpr double FixPrecision(double x) {
  return static_cast<int64_t>(x * 1000) / 1000.0;
}

}  // namespace

std::string FormatBytes(int64_t bytes) {
  if (bytes < Memsize::kKilo) {
    return fmt::format("{}b", bytes);
  }

  if (bytes < Memsize::kMega) {
    auto kb = static_cast<double>(bytes) / Memsize::kKilo;
    return fmt::format("{}kb", FixPrecision(kb));
  }

  if (bytes < Memsize::kGiga) {
    auto mb = static_cast<double>(bytes) / Memsize::kMega;
    return fmt::format("{}mb", FixPrecision(mb));
  }

  auto gb = static_cast<double>(bytes) / Memsize::kGiga;
  return fmt::format("{}gb", FixPrecision(gb));
}

}  // namespace sv
