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
 * @file implementation/ReprojectionError.hpp
 * @brief Header implementation file for the ReprojectionError class.
 * @author Stefan Leutenegger
 */

#pragma once

#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include <okvis/ceres/ReprojectionError.hpp>

namespace okvis {

namespace ceres {

template<class GEOMETRY_T>
ReprojectionError<GEOMETRY_T>::ReprojectionError()
    : cameraGeometry_(new camera_geometry_t) {
}

template<class GEOMETRY_T>
ReprojectionError<GEOMETRY_T>::ReprojectionError(
    std::shared_ptr<const camera_geometry_t> cameraGeometry, uint64_t cameraId,
    const measurement_t & measurement, const covariance_t & information) {
  setCameraId(cameraId);
  setMeasurement(measurement);
  setInformation(information);
  setCameraGeometry(cameraGeometry);
}

template<class GEOMETRY_T>
void ReprojectionError<GEOMETRY_T>::setInformation(
    const covariance_t& information) {
  information_ = information;
  covariance_ = information.inverse();

  Eigen::LLT<Eigen::Matrix2d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

template<class GEOMETRY_T>
bool ReprojectionError<GEOMETRY_T>::Evaluate(double const* const * parameters,
                                             double* residuals,
                                             double** jacobians) const {

  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

template<class GEOMETRY_T>
bool ReprojectionError<GEOMETRY_T>::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  Eigen::Map<const Eigen::Vector3d> t_WS_W(&parameters[0][0]);
  Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3],
                          parameters[0][4], parameters[0][5]);
  q_WS.normalize();

  Eigen::Map<const Eigen::Vector4d> hp_W(&parameters[1][0]);

  Eigen::Map<const Eigen::Vector3d> t_SC_S(&parameters[2][0]);
  Eigen::Quaterniond q_SC(parameters[2][6], parameters[2][3],
                          parameters[2][4], parameters[2][5]);
  q_SC.normalize();

  Eigen::Matrix3d C_SC = q_SC.toRotationMatrix();
  Eigen::Matrix3d C_CS = C_SC.transpose();
  Eigen::Matrix4d T_CS = Eigen::Matrix4d::Identity();
  T_CS.topLeftCorner<3, 3>() = C_CS;
  T_CS.topRightCorner<3, 1>() = -C_CS * t_SC_S;
  Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();
  Eigen::Matrix3d C_SW = C_WS.transpose();
  Eigen::Matrix4d T_SW = Eigen::Matrix4d::Identity();
  T_SW.topLeftCorner<3, 3>() = C_SW;
  T_SW.topRightCorner<3, 1>() = -C_SW * t_WS_W;
  Eigen::Vector4d hp_S = T_SW * hp_W;
  Eigen::Vector4d hp_C = T_CS * hp_S;

  measurement_t kp;
  Eigen::Matrix<double, 2, 4> Jh;
  Eigen::Matrix<double, 2, 4> Jh_weighted;
  if (jacobians != nullptr) {
    cameraGeometry_->projectHomogeneous(hp_C, &kp, &Jh);
    Jh_weighted = squareRootInformation_ * Jh;
  } else {
    cameraGeometry_->projectHomogeneous(hp_C, &kp);
  }

  measurement_t error = measurement_ - kp;

  measurement_t weighted_error = squareRootInformation_ * error;

  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];

  const bool valid = true;

  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      Eigen::Vector3d p = hp_W.head<3>() - t_WS_W * hp_W[3];
      Eigen::Matrix<double, 4, 6> J;
      J.setZero();
      J.topLeftCorner<3, 3>() = C_SW * hp_W[3];
      J.topRightCorner<3, 3>() = -C_SW * okvis::kinematics::crossMx(p);

      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal = Jh_weighted * T_CS * J;
      if (!valid)
        J0_minimal.setZero();

      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[0], J_lift.data());

      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J0(
          jacobians[0]);
      J0 = J0_minimal * J_lift;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[0] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > J0_minimal_mapped(
              jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }

    }
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J1(
          jacobians[1]);  
      Eigen::Matrix4d T_CW = (T_CS * T_SW);
      J1 = -Jh_weighted * T_CW;
      if (!valid)
        J1.setZero();

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[1] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J1_minimal_mapped(
              jacobiansMinimal[1]);
          Eigen::Matrix<double, 4, 3> S;
          S.setZero();
          S.topLeftCorner<3, 3>().setIdentity();
          J1_minimal_mapped = J1 * S;  
        }
      }
    }
    if (jacobians[2] != nullptr) {
      Eigen::Vector3d p = hp_S.head<3>() - t_SC_S * hp_S[3];
      Eigen::Matrix<double, 4, 6> J;
      J.setZero();
      J.topLeftCorner<3, 3>() = C_CS * hp_S[3];
      J.topRightCorner<3, 3>() = -C_CS * okvis::kinematics::crossMx(p);

      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J2_minimal;
      J2_minimal = Jh_weighted * J;
      if (!valid)
        J2_minimal.setZero();

      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[2], J_lift.data());

      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J2(
          jacobians[2]);
      J2 = J2_minimal * J_lift;

      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[2] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > J2_minimal_mapped(
              jacobiansMinimal[2]);
          J2_minimal_mapped = J2_minimal;
        }
      }
    }
  }

  return true;
}

}
}
