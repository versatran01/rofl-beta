#include "sv/util/imu.h"

#include "sv/util/eigen.h"

namespace sv {

using SO3d = Sophus::SO3d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

void ImuPreint::Update(double dt, const ImuData& imu, const ImuNoise& noise) {
  ++num_imus;
  duration += dt;

  const Vector3d acc = imu.acc - bias_hat.acc;
  const Vector3d gyr = imu.gyr - bias_hat.gyr;

  // Preintegrate
  // Rotate acc from body to first frame of the preintegration
  const Vector3d a0 = delta.q * acc;
  const Vector3d dp = delta.v * dt + 0.5 * a0 * dt * dt;
  const Vector3d dv = a0 * dt;
  const SO3d dq = SO3d::exp(gyr * dt);

  // Continuous-time error-state dynamics
  // Ft =
  // [ 0 | I | 0       |  0 | 0  ]
  // [ 0 | 0 | R*[-a]x | -R | 0  ]
  // [ 0 | 0 | [-w]x   |  0 | -I ]
  // [ 0 | 0 | 0       |  0 | 0  ]
  // [ 0 | 0 | 0       |  0 | 0  ]

  // A first-order approximation of discrete-time state transition matrix
  // F = I + Ft * dt
  // [ I | I*dt | R*[-0.5*a*dt^2]x | 0     | 0     ]
  // [ 0 | I    | R*[-a*dt]x       | -R*dt | 0     ]
  // [ 0 | 0    | Exp(-w*dt)       | 0     | -I*dt ]
  // [ 0 | 0    | 0                | I     | 0     ]
  // [ 0 | 0    | 0                | 0     | I     ]

  // Update Covariance
  const Matrix3d I3 = Matrix3d::Identity();
  const Matrix3d R = delta.q.matrix();
  Matrix9d F = Matrix9d::Identity();  // state transition matrix

  // we leave F.block<3,3>(0, 6) as 0 since it has dt^2
  F.block<3, 3>(0, 3).noalias() = I3 * dt;
  F.block<3, 3>(3, 6).noalias() = R * Hat3d(acc * (-dt));
  F.block<3, 3>(6, 6).noalias() = dq.inverse().matrix();

  P = F * P * F.transpose();

  // Add noise part
  // J_pvq_imu =
  // [ 0    | 0     ]
  // [ R*dt | 0     ]
  // [ 0    | I3*dt ]
  Matrix96d J_pvq_imu = Matrix96d::Zero();
  J_pvq_imu.block<3, 3>(3, 0) = R * dt;
  J_pvq_imu.block<3, 3>(6, 3) = I3 * dt;

  // P <- F * P * F.T + J_pvq_imu * (Q / dt) * J_pvq_imu.T
  Vector6d Q;
  Q.head<3>() = noise.acc_var;
  Q.tail<3>() = noise.gyr_var;
  // divide by dt to get discrete time noise
  P.noalias() += J_pvq_imu * (Q / dt).asDiagonal() * J_pvq_imu.transpose();

  // Jacobian
  // J <- F * J + J_pvq_bias = F * J - J_pvq_imu
  // this is because imu_nobias = imu - bias
  J = F * J - J_pvq_imu;

  // Update delta
  delta.p += dp;
  delta.v += dv;
  delta.q *= dq;
}

void ImuPreint::Reset() {
  num_imus = 0;
  duration = 0.0;
  delta = {};
  P.setZero();
  J.setZero();
}

ImuPreint::Delta ImuPreint::BiasCorrectedDelta(const ImuBias& bias_new) const {
  Vector6d bias_delta;
  bias_delta.head<3>() = bias_new.acc - bias_hat.acc;
  bias_delta.tail<3>() = bias_new.gyr - bias_hat.gyr;

  // get a copy
  Delta corr = delta;
  corr.p.noalias() += J.middleRows<3>(0) * bias_delta;
  corr.v.noalias() += J.middleRows<3>(3) * bias_delta;
  corr.q *= SO3d::exp(J.bottomRightCorner<3, 3>() * bias_delta.tail<3>());

  return corr;
}

ImuPreint::Matrix9d ImuPreint::GetInfoPvq() const {
  return P.selfadjointView<Eigen::Upper>().llt().solve(Matrix9d::Identity());
}

}  // namespace sv
