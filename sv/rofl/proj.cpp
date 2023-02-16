#include "sv/rofl/proj.h"

#include "sv/util/logging.h"

namespace sv::rofl {

Projection::Projection(cv::Size size2d, double vfov) : size2d_{size2d} {
  if (vfov <= 0) {
    // Default vfov: same horizontal and vertical resolution
    vfov = kTauD * static_cast<double>(size2d.height) / size2d.width;
  }

  CHECK_GT(size2d.width, 1);
  CHECK_GT(size2d.height, 1);
  CHECK_LE(Rad2Deg(vfov), 120.0) << "vertial fov too big";

  // max elevation is half of vertical fov
  elev_max = vfov / 2.0;
  elev_delta = vfov / (size2d.height - 1);
  azim_delta = kTauD / size2d.width;

  // elevation, from top to bottom
  elevs.resize(size2d.height);
  for (int i = 0; i < size2d.height; ++i) {
    elevs[i] = SinCosD{elev_max - i * elev_delta};
  }

  // azimuth, clockwise from +x
  azims.resize(size2d.width);
  for (int i = 0; i < size2d.width; ++i) {
    azims[i] = SinCosD{kTauD - i * azim_delta};
  }
  azims[0] = SinCosD{0.0};
}

std::string Projection::Repr() const {
  return fmt::format(
      "Projection(size={}x{}, elev_max={:.2f}[deg], "
      "elev_delta={:.4f}[deg], azim_delta={:.4f}[deg])",
      size2d_.height,
      size2d_.width,
      Rad2Deg(elev_max),
      Rad2Deg(elev_delta),
      Rad2Deg(azim_delta));
}

}  // namespace sv::rofl
