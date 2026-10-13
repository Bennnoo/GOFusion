/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file ImuError.cpp
 * @brief Source file for the ImuError class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>

#include <okvis/kinematics/operators.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ode/ode.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/PseudoInverse.hpp>

namespace okvis {

namespace ceres {

std::atomic_bool ImuError::redoPropagationAlways(false);

ImuError::ImuError(const okvis::ImuMeasurementDeque & imuMeasurements,
                   const okvis::ImuParameters & imuParameters,
                   const okvis::Time& t_0, const okvis::Time& t_1) {
  setImuMeasurements(imuMeasurements);
  setImuParameters(imuParameters);
  setT0(t_0);
  setT1(t_1);

  OKVIS_ASSERT_TRUE_DBG(Exception,
                     t_0 >= imuMeasurements.front().timeStamp,
                     "First IMU measurement included in ImuError is not old enough! "
                        << t_0 << " !>= " << imuMeasurements.front().timeStamp)
  OKVIS_ASSERT_TRUE_DBG(Exception,
                     t_1 <= imuMeasurements.back().timeStamp,
                     "Last IMU measurement included in ImuError is not new enough!"
                         << t_1 << " !<= " << imuMeasurements.back().timeStamp)

  dPdsigma_.resize(4);
  for(size_t j=0; j<4; ++j) {
    dPdsigma_.at(j).setZero();
  }
}

int ImuError::append(const okvis::kinematics::Transformation& ,
                     const okvis::SpeedAndBias & speedAndBiases,
                     const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::Time& t_1)
{

  OKVIS_ASSERT_TRUE_DBG(Exception, t1_ >= imuMeasurements.front().timeStamp,
                        "First IMU measurement included in ImuError is not old enough!")
  OKVIS_ASSERT_TRUE_DBG(Exception, t_1 <= imuMeasurements.back().timeStamp,
                        "Last IMU measurement included in ImuError is not new enough!")

  okvis::ImuMeasurementDeque::const_iterator iter = imuMeasurements.begin();
  while (iter != imuMeasurements.end()){
    if (iter->timeStamp > imuMeasurements_.back().timeStamp) {
      break;
    }
    ++iter;
  }
  imuMeasurements_.insert(imuMeasurements_.end(),iter,imuMeasurements.end());

  okvis::Time time = t1_;
  okvis::Time end = t_1;
  setT1(t_1);

  assert(imuMeasurements.front().timeStamp<=time);
  if (!(imuMeasurements.back().timeStamp >= end))
    return -1;  

  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
      it != imuMeasurements.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = t1_;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t1_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    double gyr_sat_mult = 1.0;
    double acc_sat_mult = 1.0;

    if (fabs(omega_S_0[0]) > imuParameters_.g_max
        || fabs(omega_S_0[1]) > imuParameters_.g_max
        || fabs(omega_S_0[2]) > imuParameters_.g_max
        || fabs(omega_S_1[0]) > imuParameters_.g_max
        || fabs(omega_S_1[1]) > imuParameters_.g_max
        || fabs(omega_S_1[2]) > imuParameters_.g_max) {
      gyr_sat_mult *= 100;
      LOG(WARNING)<< "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParameters_.a_max || fabs(acc_S_0[1]) > imuParameters_.a_max
        || fabs(acc_S_0[2]) > imuParameters_.a_max
        || fabs(acc_S_1[0]) > imuParameters_.a_max
        || fabs(acc_S_1[1]) > imuParameters_.a_max
        || fabs(acc_S_1[2]) > imuParameters_.a_max) {
      acc_sat_mult *= 100;
      LOG(WARNING)<< "acc saturation";
    }

    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1)
        - speedAndBiases.segment < 3 > (3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = ode::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;

    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1)
        - speedAndBiases.segment < 3 > (6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_
        + 0.5 * (C + C_1) * acc_S_true * dt;

    C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
    acc_doubleintegral_ += acc_integral_ * dt
        + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    dalpha_db_g_ += C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
        + okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_
        + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    dp_db_g_ += dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    Eigen::Matrix<double, 15, 15> F_delta =
        Eigen::Matrix<double, 15, 15>::Identity();

    F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
        acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
    F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F_delta.block<3, 3>(0, 9) = dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    F_delta.block<3, 3>(0, 12) = -C_integral_ * dt
        + 0.25 * (C + C_1) * dt * dt;
    F_delta.block<3, 3>(3, 9) = -dt * C_1;
    F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
        0.5 * (C + C_1) * acc_S_true * dt);
    F_delta.block<3, 3>(6, 9) = 0.5 * dt
        * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;

    Eigen::Matrix<double,15,15> K0 = Eigen::Matrix<double,15,15>::Zero();
    Eigen::Matrix<double,15,15> K1 = Eigen::Matrix<double,15,15>::Zero();
    Eigen::Matrix<double,15,15> K2 = Eigen::Matrix<double,15,15>::Zero();
    Eigen::Matrix<double,15,15> K3 = Eigen::Matrix<double,15,15>::Zero();
    K0.block<3,3>(3,3) = gyr_sat_mult*dt * Eigen::Matrix3d::Identity();
    K1.block<3,3>(0,0) = 0.5 * dt*dt*dt * acc_sat_mult*acc_sat_mult*acc_sat_mult
        * Eigen::Matrix3d::Identity();
    K1.block<3,3>(6,6) = acc_sat_mult*dt* Eigen::Matrix3d::Identity();
    K2.block<3,3>(9,9) = dt * Eigen::Matrix3d::Identity();
    K3.block<3,3>(12,12) = dt * Eigen::Matrix3d::Identity();
    dPdsigma_.at(0) = F_delta*dPdsigma_.at(0)*F_delta.transpose() + K0;
    dPdsigma_.at(1) = F_delta*dPdsigma_.at(1)*F_delta.transpose() + K1;
    dPdsigma_.at(2) = F_delta*dPdsigma_.at(2)*F_delta.transpose() + K2;
    dPdsigma_.at(3) = F_delta*dPdsigma_.at(3)*F_delta.transpose() + K3;

    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == t1_)
      break;

  }

  for(size_t j=0; j<4; ++j) {
    dPdsigma_.at(j) = 0.5 * dPdsigma_.at(j) + 0.5 * dPdsigma_.at(j).transpose().eval();
  }
  P_delta_ = dPdsigma_.at(0)*imuParameters_.sigma_g_c*imuParameters_.sigma_g_c;
  P_delta_ += dPdsigma_.at(1)*imuParameters_.sigma_a_c*imuParameters_.sigma_a_c;
  P_delta_ += dPdsigma_.at(2)*imuParameters_.sigma_gw_c*imuParameters_.sigma_gw_c;
  P_delta_ += dPdsigma_.at(3)*imuParameters_.sigma_aw_c*imuParameters_.sigma_aw_c;

  PseudoInverse::symmSqrtU(P_delta_, squareRootInformation_);
  information_ = squareRootInformation_.transpose()*squareRootInformation_;

  return i;
}

int ImuError::redoPreintegration(const okvis::kinematics::Transformation& ,
                                 const okvis::SpeedAndBias & speedAndBiases) const {

  okvis::Time time = t0_;
  okvis::Time end = t1_;

  OKVIS_ASSERT_TRUE_DBG(Exception, imuMeasurements_.front().timeStamp<=time,
                    imuMeasurements_.front().timeStamp << "<=" <<time)
  if (!(imuMeasurements_.back().timeStamp >= end)) {
    OKVIS_THROW_DBG(Exception, imuMeasurements_.back().timeStamp << " >= " << end)
    return -1;  
  }

  Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  C_integral_ = Eigen::Matrix3d::Zero();
  C_doubleintegral_ = Eigen::Matrix3d::Zero();
  acc_integral_ = Eigen::Vector3d::Zero();
  acc_doubleintegral_ = Eigen::Vector3d::Zero();

  cross_ = Eigen::Matrix3d::Zero();

  dalpha_db_g_ = Eigen::Matrix3d::Zero();
  dv_db_g_ = Eigen::Matrix3d::Zero();
  dp_db_g_ = Eigen::Matrix3d::Zero();

  P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

  dPdsigma_.resize(4);
  for(size_t j=0; j<4; ++j) {
    dPdsigma_.at(j).setZero();
  }

  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements_.begin();
      it != imuMeasurements_.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = t1_;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t1_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    double gyr_sat_mult = 1.0;
    double acc_sat_mult = 1.0;

    if (fabs(omega_S_0[0]) > imuParameters_.g_max
        || fabs(omega_S_0[1]) > imuParameters_.g_max
        || fabs(omega_S_0[2]) > imuParameters_.g_max
        || fabs(omega_S_1[0]) > imuParameters_.g_max
        || fabs(omega_S_1[1]) > imuParameters_.g_max
        || fabs(omega_S_1[2]) > imuParameters_.g_max) {
      gyr_sat_mult *= 100;
      LOG(WARNING)<< "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParameters_.a_max || fabs(acc_S_0[1]) > imuParameters_.a_max
        || fabs(acc_S_0[2]) > imuParameters_.a_max
        || fabs(acc_S_1[0]) > imuParameters_.a_max
        || fabs(acc_S_1[1]) > imuParameters_.a_max
        || fabs(acc_S_1[2]) > imuParameters_.a_max) {
      acc_sat_mult *= 100;
      LOG(WARNING)<< "acc saturation";
    }

    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1)
        - speedAndBiases.segment < 3 > (3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = ode::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;

    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1)
        - speedAndBiases.segment < 3 > (6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_
        + 0.5 * (C + C_1) * acc_S_true * dt;

    C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
    acc_doubleintegral_ += acc_integral_ * dt
        + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    dalpha_db_g_ += C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
        + okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_
        + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    dp_db_g_ += dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    Eigen::Matrix<double, 15, 15> F_delta =
        Eigen::Matrix<double, 15, 15>::Identity();

    F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
        acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
    F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F_delta.block<3, 3>(0, 9) = dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    F_delta.block<3, 3>(0, 12) = -C_integral_ * dt
        + 0.25 * (C + C_1) * dt * dt;
    F_delta.block<3, 3>(3, 9) = -dt * C_1;
    F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
        0.5 * (C + C_1) * acc_S_true * dt);
    F_delta.block<3, 3>(6, 9) = 0.5 * dt
        * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;

    Eigen::Matrix<double,15,15> K0 = Eigen::Matrix<double,15,15>::Zero();
    Eigen::Matrix<double,15,15> K1 = Eigen::Matrix<double,15,15>::Zero();
    Eigen::Matrix<double,15,15> K2 = Eigen::Matrix<double,15,15>::Zero();
    Eigen::Matrix<double,15,15> K3 = Eigen::Matrix<double,15,15>::Zero();
    K0.block<3,3>(3,3) = gyr_sat_mult*dt * Eigen::Matrix3d::Identity();
    K1.block<3,3>(0,0) = 0.5 * dt*dt*dt * acc_sat_mult*acc_sat_mult*acc_sat_mult
        * Eigen::Matrix3d::Identity();
    K1.block<3,3>(6,6) = acc_sat_mult*dt* Eigen::Matrix3d::Identity();
    K2.block<3,3>(9,9) = dt * Eigen::Matrix3d::Identity();
    K3.block<3,3>(12,12) = dt * Eigen::Matrix3d::Identity();
    dPdsigma_.at(0) = F_delta*dPdsigma_.at(0)*F_delta.transpose() + K0;
    dPdsigma_.at(1) = F_delta*dPdsigma_.at(1)*F_delta.transpose() + K1;
    dPdsigma_.at(2) = F_delta*dPdsigma_.at(2)*F_delta.transpose() + K2;
    dPdsigma_.at(3) = F_delta*dPdsigma_.at(3)*F_delta.transpose() + K3;

    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == t1_)
      break;

  }

  speedAndBiases_ref_ = speedAndBiases;

  for(size_t j=0; j<4; ++j) {
    dPdsigma_.at(j) = 0.5 * dPdsigma_.at(j) + 0.5 * dPdsigma_.at(j).transpose().eval();
  }
  P_delta_ = dPdsigma_.at(0)*imuParameters_.sigma_g_c*imuParameters_.sigma_g_c;
  P_delta_ += dPdsigma_.at(1)*imuParameters_.sigma_a_c*imuParameters_.sigma_a_c;
  P_delta_ += dPdsigma_.at(2)*imuParameters_.sigma_gw_c*imuParameters_.sigma_gw_c;
  P_delta_ += dPdsigma_.at(3)*imuParameters_.sigma_aw_c*imuParameters_.sigma_aw_c;

  PseudoInverse::symmSqrtU(P_delta_, squareRootInformation_);
  information_ = squareRootInformation_.transpose()*squareRootInformation_;

  return i;
}

void ImuError::syncFrom(const ImuError &other)
{
  imuParameters_ = other.imuParameters_; 
  imuMeasurements_ = other.imuMeasurements_; 
  t0_ = other.t0_; 
  t1_ = other.t1_; 
  Delta_q_ = other.Delta_q_; 
  C_integral_ = other.C_integral_; 
  C_doubleintegral_ = other.C_doubleintegral_; 
  acc_integral_ = other.acc_integral_; 
  acc_doubleintegral_ = other.acc_doubleintegral_; 
  cross_ = other.cross_; 
  dalpha_db_g_ = other.dalpha_db_g_; 
  dv_db_g_ = other.dv_db_g_; 
  dp_db_g_ = other.dp_db_g_; 
  P_delta_ = other.P_delta_;
  speedAndBiases_ref_ = other.speedAndBiases_ref_;
  redo_ = other.redo_; 
  redoCounter_ = other.redoCounter_; 
  information_ = other.information_; 
  squareRootInformation_ = other.squareRootInformation_; 
  dPdsigma_ = other.dPdsigma_;
}

void ImuError::setImuParameters(const ImuParameters &imuParameters) {

  imuParameters_ = imuParameters;

  if(dPdsigma_.size() > 0) {
    P_delta_ = dPdsigma_.at(0)*imuParameters_.sigma_g_c*imuParameters_.sigma_g_c;
    P_delta_ += dPdsigma_.at(1)*imuParameters_.sigma_a_c*imuParameters_.sigma_a_c;
    P_delta_ += dPdsigma_.at(2)*imuParameters_.sigma_gw_c*imuParameters_.sigma_gw_c;
    P_delta_ += dPdsigma_.at(3)*imuParameters_.sigma_aw_c*imuParameters_.sigma_aw_c;

    PseudoInverse::symmSqrtU(P_delta_, squareRootInformation_);
    information_ = squareRootInformation_.transpose()*squareRootInformation_;
  }
}

std::shared_ptr<ImuErrorBase> ImuError::clone() const
{
  std::shared_ptr<ImuError> clone(new ImuError());

  clone->imuParameters_ = imuParameters_; 
  clone->imuMeasurements_ = imuMeasurements_; 
  clone->t0_ = t0_; 
  clone->t1_ = t1_; 
  clone->Delta_q_ = Delta_q_; 
  clone->C_integral_ = C_integral_; 
  clone->C_doubleintegral_ = C_doubleintegral_; 
  clone->acc_integral_ = acc_integral_; 
  clone->acc_doubleintegral_ = acc_doubleintegral_; 
  clone->cross_ = cross_; 
  clone->dalpha_db_g_ = dalpha_db_g_; 
  clone->dv_db_g_ = dv_db_g_; 
  clone->dp_db_g_ = dp_db_g_; 
  clone->P_delta_ = P_delta_;
  clone->speedAndBiases_ref_ = speedAndBiases_ref_;
  clone->redo_ = redo_; 
  clone->redoCounter_ = redoCounter_; 
  clone->information_ = information_; 
  clone->squareRootInformation_ = squareRootInformation_; 
  clone->dPdsigma_ = dPdsigma_;
  return clone;
}

int ImuError::propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                          const okvis::ImuParameters & imuParams,
                          okvis::kinematics::Transformation& T_WS,
                          okvis::SpeedAndBias & speedAndBiases,
                          const okvis::Time & t_start,
                          const okvis::Time & t_end, covariance_t* covariance,
                          jacobian_t* jacobian) {

  okvis::Time time = t_start;
  okvis::Time end = t_end;

  OKVIS_ASSERT_TRUE(Exception, imuMeasurements.front().timeStamp<=time,
                        imuMeasurements.front().timeStamp << " !<= " << time)
  if (!(imuMeasurements.back().timeStamp >= end))
    return -1;  

  Eigen::Vector3d r_0 = T_WS.r();
  Eigen::Quaterniond q_WS_0 = T_WS.q();
  Eigen::Matrix3d C_WS_0 = T_WS.C();

  Eigen::Quaterniond Delta_q(1,0,0,0);
  Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d C_doubleintegral = Eigen::Matrix3d::Zero();
  Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_doubleintegral = Eigen::Vector3d::Zero();

  Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();

  Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();

  Eigen::Matrix<double,15,15> P_delta = Eigen::Matrix<double,15,15>::Zero();

  double Delta_t = 0;
  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
        it != imuMeasurements.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    double sigma_g_c = imuParams.sigma_g_c;
    double sigma_a_c = imuParams.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParams.g_max
        || fabs(omega_S_0[1]) > imuParams.g_max
        || fabs(omega_S_0[2]) > imuParams.g_max
        || fabs(omega_S_1[0]) > imuParams.g_max
        || fabs(omega_S_1[1]) > imuParams.g_max
        || fabs(omega_S_1[2]) > imuParams.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING) << "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParams.a_max || fabs(acc_S_0[1]) > imuParams.a_max
        || fabs(acc_S_0[2]) > imuParams.a_max
        || fabs(acc_S_1[0]) > imuParams.a_max
        || fabs(acc_S_1[1]) > imuParams.a_max
        || fabs(acc_S_1[2]) > imuParams.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING) << "acc saturation";
    }

    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5*(omega_S_0+omega_S_1) - speedAndBiases.segment<3>(3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = ode::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q * dq;

    const Eigen::Matrix3d C = Delta_q.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5*(acc_S_0+acc_S_1) - speedAndBiases.segment<3>(6));
    const Eigen::Matrix3d C_integral_1 = C_integral + 0.5*(C + C_1)*dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral + 0.5*(C + C_1)*acc_S_true*dt;

    C_doubleintegral += C_integral*dt + 0.25*(C + C_1)*dt*dt;
    acc_doubleintegral += acc_integral*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

    dalpha_db_g += dt*C_1;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix()*cross +
        okvis::kinematics::rightJacobian(omega_S_true*dt)*dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g + 0.5*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);
    dp_db_g += dt*dv_db_g + 0.25*dt*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);

    if (covariance) {
      Eigen::Matrix<double,15,15> F_delta = Eigen::Matrix<double,15,15>::Identity();

      F_delta.block<3,3>(0,3) = -okvis::kinematics::crossMx(
            acc_integral*dt + 0.25*(C + C_1)*acc_S_true*dt*dt);
      F_delta.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*dt;
      F_delta.block<3,3>(0,9) = dt*dv_db_g + 0.25*dt*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);
      F_delta.block<3,3>(0,12) = -C_integral*dt + 0.25*(C + C_1)*dt*dt;
      F_delta.block<3,3>(3,9) = -dt*C_1;
      F_delta.block<3,3>(6,3) = -okvis::kinematics::crossMx(0.5*(C + C_1)*acc_S_true*dt);
      F_delta.block<3,3>(6,9) = 0.5*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);
      F_delta.block<3,3>(6,12) = -0.5*(C + C_1)*dt;

      P_delta = F_delta*P_delta*F_delta.transpose();

      const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
      P_delta(3,3) += sigma2_dalpha;
      P_delta(4,4) += sigma2_dalpha;
      P_delta(5,5) += sigma2_dalpha;
      const double sigma2_v = dt * sigma_a_c * imuParams.sigma_a_c;
      P_delta(6,6) += sigma2_v;
      P_delta(7,7) += sigma2_v;
      P_delta(8,8) += sigma2_v;
      const double sigma2_p = 0.5*dt*dt*sigma2_v;
      P_delta(0,0) += sigma2_p;
      P_delta(1,1) += sigma2_p;
      P_delta(2,2) += sigma2_p;
      const double sigma2_b_g = dt * imuParams.sigma_gw_c * imuParams.sigma_gw_c;
      P_delta(9,9)   += sigma2_b_g;
      P_delta(10,10) += sigma2_b_g;
      P_delta(11,11) += sigma2_b_g;
      const double sigma2_b_a = dt * imuParams.sigma_aw_c * imuParams.sigma_aw_c;
      P_delta(12,12) += sigma2_b_a;
      P_delta(13,13) += sigma2_b_a;
      P_delta(14,14) += sigma2_b_a;
    }

    Delta_q = Delta_q_1;
    C_integral = C_integral_1;
    acc_integral = acc_integral_1;
    cross = cross_1;
    dv_db_g = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == t_end)
      break;

  }

  const Eigen::Vector3d g_W = imuParams.g * Eigen::Vector3d(0, 0, 6371009).normalized();
  T_WS.set(r_0+speedAndBiases.head<3>()*Delta_t
             + C_WS_0*(acc_doubleintegral)
             - 0.5*g_W*Delta_t*Delta_t,
             q_WS_0*Delta_q);
  speedAndBiases.head<3>() += C_WS_0*(acc_integral)
      -g_W*Delta_t;

  if (jacobian) {
    Eigen::Matrix<double,15,15> & F = *jacobian;
    F.setIdentity(); 
    F.block<3,3>(0,3) = -okvis::kinematics::crossMx(C_WS_0*acc_doubleintegral);
    F.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*Delta_t;
    F.block<3,3>(0,9) = C_WS_0*dp_db_g;
    F.block<3,3>(0,12) = -C_WS_0*C_doubleintegral;
    F.block<3,3>(3,9) = -C_WS_0*dalpha_db_g;
    F.block<3,3>(6,3) = -okvis::kinematics::crossMx(C_WS_0*acc_integral);
    F.block<3,3>(6,9) = C_WS_0*dv_db_g;
    F.block<3,3>(6,12) = -C_WS_0*C_integral;
  }

  if (covariance) {
    Eigen::Matrix<double,15,15> & P = *covariance;

    Eigen::Matrix<double,15,15> T = Eigen::Matrix<double,15,15>::Identity();
    T.topLeftCorner<3,3>() = C_WS_0;
    T.block<3,3>(3,3) = C_WS_0;
    T.block<3,3>(6,6) = C_WS_0;
    P = T * P_delta * T.transpose();
  }
  return i;
}

bool ImuError::initPose(const ImuMeasurementDeque &imuMeasurements,
                        kinematics::Transformation &T_WS) {

  T_WS.setIdentity();

  if (imuMeasurements.size() == 0) return false;

  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it < imuMeasurements.end(); ++it) {
    acc_B += it->measurement.accelerometers;
  }
  acc_B /= double(imuMeasurements.size());
  Eigen::Vector3d e_acc = acc_B.normalized();

  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);

  return true;
}

bool ImuError::Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool ImuError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const {
  bool success = true;

  Eigen::Map<Eigen::Matrix<double, 15, 1> > weighted_error(residuals);

  const okvis::kinematics::Transformation T_WS_0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(
        parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]).normalized());

  const okvis::kinematics::Transformation T_WS_1(
      Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
      Eigen::Quaterniond(
        parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]).normalized());

  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (int i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;

  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    Delta_b = speedAndBiases_0.tail<6>()
          - speedAndBiases_ref_.tail<6>();
    redo_ = redo_ || (Delta_b.head<3>().norm() > 0.0003);
    if ((redo_ && ((imuMeasurements_.size() < 50) || redoPropagationAlways)) || redoCounter_==0) {
      const int steps = redoPreintegration(T_WS_0, speedAndBiases_0);
      if(steps<=0) {
        for(const auto & m : imuMeasurements_)
          std::cout << m.timeStamp << std::endl;
      }
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            steps > 0,
                            "IMU error term computation failed: 0 steps integrated: "
                              << imuMeasurements_.front().timeStamp << " - "
                              << imuMeasurements_.back().timeStamp << " , "
                              << imuMeasurements_.size())
      if (steps == 0) {

        LOG(WARNING) << "IMU error term computation failed: 0 steps integrated -- disable.";
        success = false;
      }

      redoCounter_++;
      Delta_b.setZero();
      redo_ = false;
    }
  }

  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);

    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    Eigen::Matrix<double,15,15> F0 =
        Eigen::Matrix<double,15,15>::Identity(); 
    const Eigen::Vector3d delta_p_est_W =
        T_WS_0.r() - T_WS_1.r() + speedAndBiases_0.head<3>()*Delta_t - 0.5*g_W*Delta_t*Delta_t;
    const Eigen::Vector3d delta_v_est_W =
        speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>() - g_W*Delta_t;
    const Eigen::Quaterniond Dq =
        okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b.head<3>())*Delta_q_;
    F0.block<3,3>(0,0) = C_S0_W;
    F0.block<3,3>(0,3) = C_S0_W * okvis::kinematics::crossMx(delta_p_est_W);
    F0.block<3,3>(0,6) = C_S0_W * Eigen::Matrix3d::Identity()*Delta_t;
    F0.block<3,3>(0,9) = dp_db_g_;
    F0.block<3,3>(0,12) = -C_doubleintegral_;
    F0.block<3,3>(3,3) = (okvis::kinematics::plus(Dq*T_WS_1.q().inverse()) *
        okvis::kinematics::oplus(T_WS_0.q())).topLeftCorner<3,3>();
    F0.block<3,3>(3,9) = (okvis::kinematics::oplus(T_WS_1.q().inverse()*T_WS_0.q())*
        okvis::kinematics::oplus(Dq)).topLeftCorner<3,3>()*(-dalpha_db_g_);
    F0.block<3,3>(6,3) = C_S0_W * okvis::kinematics::crossMx(delta_v_est_W);
    F0.block<3,3>(6,6) = C_S0_W;
    F0.block<3,3>(6,9) = dv_db_g_;
    F0.block<3,3>(6,12) = -C_integral_;

    Eigen::Matrix<double,15,15> F1 =
        -Eigen::Matrix<double,15,15>::Identity(); 
    F1.block<3,3>(0,0) = -C_S0_W;
    F1.block<3,3>(3,3) = -(okvis::kinematics::plus(Dq) *
        okvis::kinematics::oplus(T_WS_0.q()) *
        okvis::kinematics::plus(T_WS_1.q().inverse())).topLeftCorner<3,3>();
    F1.block<3,3>(6,6) = -C_S0_W;

    Eigen::Matrix<double, 15, 1> error;
    error.segment<3>(0) =  C_S0_W * delta_p_est_W + acc_doubleintegral_
        + F0.block<3,6>(0,9)*Delta_b;
    error.segment<3>(3) = 2*(Dq*(T_WS_1.q().inverse()*T_WS_0.q())).vec();
    error.segment<3>(6) = C_S0_W * delta_v_est_W + acc_integral_ + F0.block<3,6>(6,9)*Delta_b;
    error.tail<6>() = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();
    if (!success) {
      error.setZero(); 
    }

    weighted_error = squareRootInformation_ * error;

    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {

        Eigen::Matrix<double, 15, 6> J0_minimal = squareRootInformation_
            * F0.block<15, 6>(0, 0);

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[0], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J0(
            jacobians[0]);
        J0 = J0_minimal * J_lift;

        if (jacobiansMinimal != nullptr) {
          if (jacobiansMinimal[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J0_minimal_mapped(
                jacobiansMinimal[0]);
            J0_minimal_mapped = J0_minimal;
            if (!success) {
              J0_minimal_mapped.setZero();
            }
          }
        }
      }
      if (jacobians[1] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J1(
            jacobians[1]);
        J1 = squareRootInformation_ * F0.block<15, 9>(0, 6);

        if (jacobiansMinimal != nullptr) {
          if (jacobiansMinimal[1] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J1_minimal_mapped(
                jacobiansMinimal[1]);
            J1_minimal_mapped = J1;
            if (!success) {
              J1_minimal_mapped.setZero();
            }
          }
        }
      }
      if (jacobians[2] != nullptr) {

        Eigen::Matrix<double, 15, 6> J2_minimal = squareRootInformation_
                    * F1.block<15, 6>(0, 0);

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[2], J_lift.data());

        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J2(
            jacobians[2]);
        J2 = J2_minimal * J_lift;

        if (jacobiansMinimal != nullptr) {
          if (jacobiansMinimal[2] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J2_minimal_mapped(
                jacobiansMinimal[2]);
            J2_minimal_mapped = J2_minimal;
            if (!success) {
              J2_minimal_mapped.setZero();
            }
          }
        }
      }
      if (jacobians[3] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J3(jacobians[3]);
        J3 = squareRootInformation_ * F1.block<15, 9>(0, 6);

        if (jacobiansMinimal != nullptr) {
          if (jacobiansMinimal[3] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J3_minimal_mapped(
                jacobiansMinimal[3]);
            J3_minimal_mapped = J3;
            if (!success) {
              J3_minimal_mapped.setZero();
            }
          }
        }
      }
    }
  }
  return true;
}

bool ImuError::EvaluateWithSigmaGradientAndHessian(const double * const *parameters, double *cost,
                                                   double *gradient, double *hessian) const
{

  const okvis::kinematics::Transformation T_WS_0(
        Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));

  const okvis::kinematics::Transformation T_WS_1(
      Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
      Eigen::Quaterniond(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]));

  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (int i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;

  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    Delta_b = speedAndBiases_0.tail<6>()
          - speedAndBiases_ref_.tail<6>();
  }

  Eigen::Matrix<double, 15, 1> error;
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);

    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    Eigen::Matrix<double,15,15> F0 =
        Eigen::Matrix<double,15,15>::Identity(); 
    const Eigen::Vector3d delta_p_est_W =
        T_WS_0.r() - T_WS_1.r() + speedAndBiases_0.head<3>()*Delta_t - 0.5*g_W*Delta_t*Delta_t;
    const Eigen::Vector3d delta_v_est_W =
        speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>() - g_W*Delta_t;
    const Eigen::Quaterniond Dq =
        okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b.head<3>())*Delta_q_;
    F0.block<3,3>(0,0) = C_S0_W;
    F0.block<3,3>(0,3) = C_S0_W * okvis::kinematics::crossMx(delta_p_est_W);
    F0.block<3,3>(0,6) = C_S0_W * Eigen::Matrix3d::Identity()*Delta_t;
    F0.block<3,3>(0,9) = dp_db_g_;
    F0.block<3,3>(0,12) = -C_doubleintegral_;
    F0.block<3,3>(3,3) = (okvis::kinematics::plus(Dq*T_WS_1.q().inverse()) *
        okvis::kinematics::oplus(T_WS_0.q())).topLeftCorner<3,3>();
    F0.block<3,3>(3,9) = (okvis::kinematics::oplus(T_WS_1.q().inverse()*T_WS_0.q())*
        okvis::kinematics::oplus(Dq)).topLeftCorner<3,3>()*(-dalpha_db_g_);
    F0.block<3,3>(6,3) = C_S0_W * okvis::kinematics::crossMx(delta_v_est_W);
    F0.block<3,3>(6,6) = C_S0_W;
    F0.block<3,3>(6,9) = dv_db_g_;
    F0.block<3,3>(6,12) = -C_integral_;

    Eigen::Matrix<double,15,15> F1 =
        -Eigen::Matrix<double,15,15>::Identity(); 
    F1.block<3,3>(0,0) = -C_S0_W;
    F1.block<3,3>(3,3) = -(okvis::kinematics::plus(Dq) *
        okvis::kinematics::oplus(T_WS_0.q()) *
        okvis::kinematics::plus(T_WS_1.q().inverse())).topLeftCorner<3,3>();
    F1.block<3,3>(6,6) = -C_S0_W;

    error.segment<3>(0) =  C_S0_W * delta_p_est_W + acc_doubleintegral_
        + F0.block<3,6>(0,9)*Delta_b;
    error.segment<3>(3) = 2*(Dq*(T_WS_1.q().inverse()*T_WS_0.q())).vec();
    error.segment<3>(6) = C_S0_W * delta_v_est_W + acc_integral_ + F0.block<3,6>(6,9)*Delta_b;
    error.tail<6>() = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();
  }

  *cost = 0.5*(error.transpose()*information_*error + log(P_delta_.determinant()));

  Eigen::Map<Eigen::Matrix<double, 4, 1>> gradientVec(gradient);
  Eigen::Map<Eigen::Matrix<double, 4, 4>> hessianMat(hessian);
  for(size_t j = 0; j<4; ++j) {

    Eigen::Matrix<double, 15, 1> inf_e = information_*error;

    Eigen::Matrix<double, 15, 15> Y = information_*dPdsigma_.at(j);
    gradientVec[int(j)] = 0.5*(-inf_e.transpose()*dPdsigma_.at(j)*inf_e + Y.trace());

    Eigen::Matrix<double, 15, 15> Ydash = -Y*information_*dPdsigma_.at(j);
    hessianMat(int(j),int(j)) = 0.5*(2*(inf_e.transpose()*dPdsigma_.at(j))*information_
                                     *(dPdsigma_.at(j)*inf_e) + Ydash.trace());
    for(size_t k = 0; k<j; ++k) {

      Ydash = -Y*information_*dPdsigma_.at(k);
      const double t0 = ((inf_e.transpose()*dPdsigma_.at(j))*information_*(dPdsigma_.at(k)*inf_e));
      const double t1 = ((inf_e.transpose()*dPdsigma_.at(k))*information_*(dPdsigma_.at(j)*inf_e));
      hessianMat(int(j),int(k)) = 0.5*(t0 + t1 + Ydash.trace());
      hessianMat(int(k),int(j)) = hessianMat(int(j),int(k));
    }
  }

  return true;
}

PseudoImuError::PseudoImuError(const Time &t_0, const Time &t_1)
{
  t0_ = t_0; 
  t1_ = t_1; 
}

void PseudoImuError::syncFrom(const PseudoImuError &other)
{
  t0_ = other.t0_; 
  t1_ = other.t1_; 
}

std::shared_ptr<ImuErrorBase> PseudoImuError::clone() const
{
  std::shared_ptr<PseudoImuError> clone(new PseudoImuError());

  clone->t0_ = t0_; 
  clone->t1_ = t1_; 
  return clone;
}

bool PseudoImuError::Evaluate(double const *const *parameters,
                              double *residuals,
                              double **jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool PseudoImuError::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                  double *residuals,
                                                  double **jacobians,
                                                  double **jacobiansMinimal) const
{

  const okvis::kinematics::Transformation
    T_WS_0(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
           Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5])
             .normalized());

  const okvis::kinematics::Transformation
    T_WS_1(Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
           Eigen::Quaterniond(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5])
             .normalized());

  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (int i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Vector3d dr = speedAndBiases_0.head<3>() * Delta_t;
  kinematics::Transformation T_WS_1_predicted(T_WS_0.r() + dr, T_WS_0.q());

  Eigen::Matrix<double, 15, 1> error;
  error.head<3>() = C_S0_W * (T_WS_1_predicted.r() - T_WS_1.r());
  error.segment<3>(3) = 2.0 * ((T_WS_1.q().inverse() * T_WS_0.q()).coeffs().head<3>());
  error.segment<3>(6) = C_S0_W * (speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>());
  error.segment<6>(9) = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();
  Eigen::Map<Eigen::Matrix<double, 15, 1>> weighted_error(residuals);
  information_t squareRootInformation = information_t::Identity(); 
  squareRootInformation.block<3, 3>(0, 0) *= 1.0 / sqrt(Delta_t);
  squareRootInformation.block<3, 3>(3, 3) *= 1.0 / sqrt(Delta_t);
  squareRootInformation.block<3, 3>(6, 6) *= 0.10 / sqrt(Delta_t);
  squareRootInformation.block<3, 3>(9, 9) *= 1.0 / sqrt(Delta_t);
  squareRootInformation.block<3, 3>(12, 12) *= 1.0 / sqrt(Delta_t);
  weighted_error = squareRootInformation * error;

  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {

      Eigen::Matrix<double, 15, 6> Jp0 = Eigen::Matrix<double, 15, 6>::Zero();
      Jp0.block<3, 3>(0, 0) = C_S0_W;
      Jp0.block<3, 3>(0, 3) = C_S0_W * kinematics::crossMx(T_WS_1_predicted.r() - T_WS_1.r());
      Jp0.block<3, 3>(3, 3)
        = kinematics::plus(T_WS_1.q().inverse() * T_WS_0.q()).topLeftCorner<3, 3>() * C_S0_W;
      Jp0.block<3, 3>(6, 3) = C_S0_W
                              * kinematics::crossMx(speedAndBiases_0.head<3>()
                                                    - speedAndBiases_1.head<3>());
      Eigen::Matrix<double, 15, 6> J0_minimal = squareRootInformation * Jp0;

      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[0], J_lift.data());

      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0 = J0_minimal * J_lift;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[0] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J0_minimal_mapped(
            jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J1(jacobians[1]);
      Eigen::Matrix<double, 15, 9> Jsb0 = Eigen::Matrix<double, 15, 9>::Zero();
      Jsb0.block<3, 3>(0, 0) = Delta_t * C_S0_W;
      Jsb0.block<3, 3>(6, 0) = C_S0_W;
      Jsb0.bottomRightCorner<6, 6>() = Eigen::Matrix<double, 6, 6>::Identity();
      J1 = squareRootInformation * Jsb0;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[1] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J1_minimal_mapped(
            jacobiansMinimal[1]);
          J1_minimal_mapped = J1;
        }
      }
    }
    if (jacobians[2] != nullptr) {

      Eigen::Matrix<double, 15, 6> Jp1 = Eigen::Matrix<double, 15, 6>::Zero();
      Jp1.block<3, 3>(0, 0) = -C_S0_W;
      Jp1.block<3, 3>(3, 3)
        = -kinematics::plus(T_WS_1.q().inverse() * T_WS_0.q()).topLeftCorner<3, 3>() * C_S0_W;
      Eigen::Matrix<double, 15, 6> J2_minimal = squareRootInformation * Jp1;

      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[2], J_lift.data());

      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J2(
        jacobians[2]);
      J2 = J2_minimal * J_lift;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[2] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J2_minimal_mapped(
            jacobiansMinimal[2]);
          J2_minimal_mapped = J2_minimal;
        }
      }
    }
    if (jacobians[3] != nullptr) {
      Eigen::Matrix<double, 15, 9> Jsb1 = Eigen::Matrix<double, 15, 9>::Zero();
      Jsb1.block<3, 3>(6, 0) = -C_S0_W;
      Jsb1.bottomRightCorner<6, 6>() = -Eigen::Matrix<double, 6, 6>::Identity();
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J3(jacobians[3]);
      J3 = squareRootInformation * Jsb1;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[3] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J3_minimal_mapped(
            jacobiansMinimal[3]);
          J3_minimal_mapped = J3;
        }
      }
    }
  }

  return true;
}

}  
}  
