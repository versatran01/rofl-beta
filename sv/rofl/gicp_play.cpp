#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "sv/rofl/gicp.h"
#include "sv/util/summary.h"

ABSL_FLAG(bool, tbb, false, "use tbb");
ABSL_FLAG(int, rep, 1, "rep");

namespace sv::rofl {

void Play() {
  TimerSummary ts{"rofl/gicp"};

  const int tbb = static_cast<int>(absl::GetFlag(FLAGS_tbb));
  const int rep = absl::GetFlag(FLAGS_rep);
  LOG(INFO) << "tbb: " << tbb;
  LOG(INFO) << "rep: " << rep;

  // Make a test sweep
  auto sweep = MakeTestSweep({1024, 64}, 20.0);
  for (int i = 0; i < rep; ++i) {
    auto _ = ts.Scoped("RangeGrad");
    sweep.CalcRangeGrad2(tbb);
  }
  LOG(INFO) << sweep.Repr();

  // Select from grid
  SweepGrid grid;
  grid.Allocate(sweep.size2d());

  int n_sel{};
  for (int i = 0; i < rep; ++i) {
    auto _ = ts.Scoped("GridSelect");
    n_sel = grid.Select(sweep, tbb);
  }
  LOG(INFO) << "n_sel: " << n_sel;

  // Make a test pano
  const Projection lidar({1024, 256});
  LOG(INFO) << lidar.Repr();
  auto pano = MakeTestPano(lidar.size2d(), 20.0);
  pano.set_id(0);
  LOG(INFO) << pano.Repr();

  LOG(INFO) << ts.ReportAll(true);
}

}  // namespace sv::rofl

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  sv::rofl::Play();
}
