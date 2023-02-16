#include "sv/rofl/pano.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

TEST(Pano, Fuse) {
  PanoData data;
  PanoCfg cfg;

  // Add first one
  Fuse(data, cfg, 10.0, 0, 0);
  EXPECT_EQ(data.info, cfg.max_info / 2);
  EXPECT_EQ(data.GetRange(), 10.0);

  // Add another good one
  Fuse(data, cfg, 10.0, 0, 0);
  EXPECT_EQ(data.info, cfg.max_info / 2 + 1);
  EXPECT_EQ(data.GetRange(), 10.0);

  // Add a bad one
  Fuse(data, cfg, 12.0, 0, 0);
  EXPECT_EQ(data.info, cfg.max_info / 2);
  EXPECT_EQ(data.GetRange(), 10.0);
}

namespace bm = benchmark;

void BM_PanoAdd(bm::State& state) {
  Projection proj({1024, 256});
  DepthPano pano(proj.size2d());

  const auto sweep = MakeTestSweep({1024, 64});
  const auto gsize = static_cast<int>(state.range(0));

  for (auto _ : state) {
    pano.AddSweep(sweep, proj, sweep.span(), gsize);
    bm::DoNotOptimize(pano);
    bm::ClobberMemory();
  }
}
BENCHMARK(BM_PanoAdd)->Arg(0)->Arg(1);

}  // namespace
}  // namespace sv::rofl
