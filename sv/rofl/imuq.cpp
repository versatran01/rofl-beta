#include "sv/rofl/imuq.h"

#include "sv/util/logging.h"

namespace sv::rofl {

void ImuqCfg::Check() const {
  CHECK_GT(bufsize, 0);
  CHECK_GT(rate, 0);
}

std::string ImuqCfg::Repr() const {
  return fmt::format(
      "ImuqCfg(bufsize={}, rate={}, acc_sigma={}, gyr_sigma={}, "
      "acc_bias_sigma={}, gyr_bias_sigma={})",
      bufsize,
      rate,
      acc_sigma,
      gyr_sigma,
      acc_bias_sigma,
      gyr_bias_sigma);
}

ImuQueue::ImuQueue(const ImuqCfg& cfg)
    : noise{cfg.acc_sigma,
            cfg.gyr_sigma,
            cfg.acc_bias_sigma,
            cfg.gyr_bias_sigma},
      buf(cfg.bufsize) {}

std::string ImuQueue::Repr() const {
  return fmt::format("ImuQueue(size={}, capacity={})", size(), capacity());
}

bool ImuQueue::Add(const ImuData& imu) {
  if (imu.HasNan()) return false;
  if (!empty()) CHECK_GT(imu.time, buf.back().time);

  buf.push_back(imu);
  return true;
}

int ImuQueue::FindAfter(double t) const {
  CHECK(!empty());
  return ImuIndexAfterTime(buf, t);
}

int ImuQueue::FindBefore(double t) const {
  CHECK(!empty());
  return ImuIndexBeforeTime(buf, t);
}

MeanCovar3d ImuQueue::CalcAccMean() const {
  MeanCovar3d mc;
  for (const auto& imu : buf) {
    mc.Add(imu.acc);
  }
  return mc;
}

MeanCovar3d ImuQueue::CalcGyrMean() const {
  MeanCovar3d mc;
  for (const auto& imu : buf) {
    mc.Add(imu.gyr);
  }
  return mc;
}

int ImuIndexAfterTime(const ImuBuffer& buf, double t) {
  // NOTE (rofl): one could probably use upper_bound for this, but because we
  // know that imu is around 100hz, which means we will have around 10 imus with
  // in one sweep. Thus it is always faster to just search backwards, especially
  // when the buffer size is big.

  // Start with 1 past the last one
  auto i = static_cast<int>(buf.size());
  for (; i > 0; --i) {
    // If the one before i is earlier in time, then stop
    if (buf.at(i - 1).time <= t) break;
  }

  return i;
}

int ImuIndexBeforeTime(const ImuBuffer& buf, double t) {
  // Start with the last one
  auto i = static_cast<int>(buf.size()) - 1;
  for (; i >= 0; --i) {
    if (buf.at(i).time <= t) break;
  }

  return i;
}

ImuQueue MakeTestImuq(int size) {
  ImuqCfg cfg;
  cfg.bufsize = size;
  cfg.rate = 1.0;

  ImuQueue imuq(cfg);
  for (int i = 0; i < size; ++i) {
    ImuData d;
    d.time = i + 1;
    imuq.Add(d);
  }

  return imuq;
}

}  // namespace sv::rofl
