#include "sv/rofl/traj.h"

#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::rofl {

using SO3f = Sophus::SO3f;
using SE3f = Sophus::SE3f;
using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;

std::string NavState::Repr() const {
  const auto& q = rot.unit_quaternion();
  return fmt::format("NavState(t={}, quat=[{}], pos=[{}], vel=[{}]",
                     time,
                     rot.unit_quaternion().coeffs().transpose(),
                     pos.transpose(),
                     vel.transpose());
}

std::string TrajCfg::Repr() const {
  return fmt::format(
      "TrajCfg(use_acc={}, update_bias={})", use_acc, update_bias);
}

std::string Trajectory::Repr() const {
  return fmt::format(
      "size={}, segs={}, cfg={}", size(), segments(), cfg_.Repr());
}

SE3d Trajectory::GetTfOdomLidar() const {
  // tf_o_l = tf_o_i * tf_i_l
  const SE3d tf_o_i{last().rot, last().pos};
  return tf_o_i * tf_i_l_;
}

SO3d Trajectory::GetRotWorldOdom() const {
  CHECK_GT(gravity_i_.squaredNorm(), 0) << "Gravity not set";
  return Sophus::SO3d(
      Eigen::Quaterniond::FromTwoVectors(gravity_i_, Eigen::Vector3d::UnitZ()));
}

IntPair Trajectory::PredictLast(const ImuQueue& imuq,
                                const TimePair& time_range,
                                int num_segs) {
  CHECK(!empty());
  CHECK(!imuq.empty());
  CHECK_GT(num_segs, 0);
  CHECK_LE(num_segs, segments());
  CHECK_GT(time_range.second, time_range.first);

  const auto dt = (time_range.second - time_range.first) / num_segs;
  const int ist0 = segments() - num_segs;

  // Update time of the starting state
  states_.at(ist0).time = time_range.first;

  // Find first imu right after t0 (>)
  const int ibuf0 = imuq.FindAfter(time_range.first);
  // TODO (rofl): consider change this to warning istead of check
  CHECK_GT(ibuf0, 0) << fmt::format(
      "t0 {} before time of first imu {}, t_imu_front - t0 = {}",
      time_range.first,
      imuq.first().time,
      imuq.first().time - time_range.first);
  CHECK_LT(ibuf0, imuq.size())
      << fmt::format("t0 {} after time of last imu {}, t0 - t_imu_back = {}",
                     time_range.first,
                     imuq.last().time,
                     time_range.first - imuq.last().time);

  int ibuf = ibuf0;
  auto imu0 = imuq.DebiasedAt(ibuf - 1);
  auto imu1 = imuq.DebiasedAt(ibuf);

  for (int ist = ist0 + 1; ist < length(); ++ist) {
    const auto& prev = states_.at(ist - 1);
    auto& curr = states_.at(ist);

    // TODO (rofl): for now we only integrate rotation
    curr.vel = prev.vel;
    curr.pos = prev.pos;
    curr.rot = prev.rot;
    if (cfg_.motion_comp) {
      curr.pos += prev.vel * dt;
      curr.rot *= DeltaRot(prev.time, imu0, imu1, dt);
    }
    curr.time = prev.time + dt;

    // Scan imu to find the next imu that
    while (imu1.time < curr.time && ibuf + 1 < imuq.size()) {
      ++ibuf;
      imu0 = imu1;
      imu1 = imuq.DebiasedAt(ibuf);
    }
  }

  return {ibuf0 - 1, ibuf};
}

IntPair Trajectory::PredictNew(const ImuQueue& imuq,
                               const TimePair& time_range,
                               int num_segs) {
  // Pop oldest states (rotate) so that we can predict forward n states
  PopOldest(num_segs);
  return PredictLast(imuq, time_range, num_segs);
}

IntPair Trajectory::PredictFull(const ImuQueue& imuq) {
  CHECK(ok());
  const auto t0 = first().time;
  const auto t1 = last().time;
  return PredictLast(imuq, {t0, t1}, segments());
}

void Trajectory::PopOldest(int n) {
  CHECK_LE(0, n);
  CHECK_LT(n, size());
  std::rotate(states_.begin(), states_.begin() + n, states_.end());
}

void Trajectory::Interp(std::vector<SE3f>& tfs,
                        int col_end,
                        double offset,
                        int gsize) const {
  CHECK(!empty());
  CHECK(!tfs.empty());

  const int num_segs = segments();
  const auto num_cols = static_cast<int>(tfs.size());

  // Make sure we can evenly divide sweep cols by segments
  CHECK_LE(col_end, num_cols);
  CHECK_EQ(num_cols % num_segs, 0);

  // cols per traj segment
  const int seg_cols = num_cols / num_segs;
  const int seg_end = col_end / seg_cols;

  // for each segment in sweep
  ParallelFor({0, num_segs, gsize}, [&](int seg) {
    // Note that the starting point of traj is where sweep buffer ends, so we
    // need to offset by seg_end to find the corresponding traj segment
    int i = seg - seg_end;
    if (i < 0) i += num_segs;

    const auto& st0 = StateAt(i);
    const auto& st1 = StateAt(i + 1);

    const auto dr = (st0.rot.inverse() * st1.rot).log();
    const auto dp = (st1.pos - st0.pos).eval();

    SE3d tf_o_i;
    for (int j = 0; j < seg_cols; ++j) {
      const int c = seg * seg_cols + j;
      const auto l = (j + offset) / seg_cols;
      tf_o_i.so3() = st0.rot * SO3d::exp(l * dr);
      tf_o_i.translation() = st0.pos + l * dp;
      tfs.at(c) = (tf_o_i * tf_i_l_).cast<float>();
    }
  });
}

void Trajectory::Interp2(std::vector<SE3f>& tfs,
                         int col_end,
                         double offset,
                         int gsize) const {
  CHECK(!empty());
  CHECK(!tfs.empty());

  const int num_cols = tfs.size();
  const int num_segs = segments();

  // Make sure we can evenly divide sweep cols by segments
  CHECK_LE(col_end, num_cols);
  CHECK_EQ(num_cols % num_segs, 0);

  // cols per traj segment
  const int seg_cols = num_cols / num_segs;
  const int seg_end = col_end / seg_cols;

  // for each segment in sweep
  ParallelFor({0, num_segs, gsize}, [&](int seg) {
    // Note that the starting point of traj is where sweep buffer ends, so we
    // need to offset by seg_end to find the corresponding traj segment
    int i = seg - seg_end;
    if (i < 0) i += num_segs;

    // T_odom_imu
    const auto& st0 = StateAt(i);
    const auto& st1 = StateAt(i + 1);

    // T_odom_lidar = T_odom_imu * T_imu_lidar
    const auto tf0 = (st0.tf() * tf_i_l_).cast<float>();
    const auto tf1 = (st1.tf() * tf_i_l_).cast<float>();

    const auto dr = (tf0.so3().inverse() * tf1.so3()).log();
    const auto dp = (tf1.translation() - tf0.translation()).eval();

    for (int j = 0; j < seg_cols; ++j) {
      const int c = seg * seg_cols + j;
      const auto l = (j + offset) / seg_cols;
      auto& tf = tfs.at(c);
      tf.so3() = (tf0.so3() * SO3f::exp(l * dr));
      tf.translation() = (tf0.translation() + l * dp);
    }
  });
}

void Trajectory::UpdateFirst(const SE3d& dtf_l) {
  CHECK(!empty());
  CHECK(ok());

  const auto dt = duration();
  CHECK_GT(dt, 0) << fmt::format(
      "t0: {}, t1: {}", states_.front().time, states_.back().time);
  auto& st = states_.front();

  // transform dtf from lidar to imu frame
  const auto dtf_i = tf_i_l_ * dtf_l * tf_i_l_.inverse();
  const auto dpos = st.rot * dtf_i.translation();

  st.pos += dpos;
  st.vel += dpos / dt * 0.5;
  st.rot *= dtf_i.so3();
}

const NavState& Trajectory::first() const {
  CHECK(!empty());
  return states_.front();
}

const NavState& Trajectory::last() const {
  CHECK(!empty());
  return states_.back();
}

size_t Trajectory::Allocate(int grid_cols) {
  states_.resize(grid_cols + 1);
  return states_.size() * sizeof(SE3f);
}

SO3d DeltaRot(double time,
              const ImuData& imu0,
              const ImuData& imu1,
              double dt) {
  const auto s = InterpTime(time, imu0.time, imu1.time);
  const Eigen::Vector3d omg = (1 - s) * imu0.gyr + s * imu1.gyr;
  return SO3d::exp(omg * dt);
}

}  // namespace sv::rofl
