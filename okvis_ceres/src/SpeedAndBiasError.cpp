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
 * @file SpeedAndBiasError.cpp
 * @brief Source file for the SpeedAndBiasError class.
 * @author Stefan Leutenegger
 */

#include <okvis/ceres/SpeedAndBiasError.hpp>

namespace okvis {

namespace ceres {

SpeedAndBiasError::SpeedAndBiasError(const okvis::SpeedAndBias & measurement,
                                     const information_t & information) {
  setMeasurement(measurement);
  setInformation(information);
}

SpeedAndBiasError::SpeedAndBiasError(const okvis::SpeedAndBias& measurement,
                                     double speedVariance,
                                     double gyrBiasVariance,
                                     double accBiasVariance) {
  setMeasurement(measurement);

  information_t information;
  information.setZero();
  information.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / speedVariance;
  information.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1.0
      / gyrBiasVariance;
  information.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / accBiasVariance;

  setInformation(information);
}

void SpeedAndBiasError::setInformation(const information_t & information) {
  information_ = information;
  covariance_ = information.inverse();

  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

bool SpeedAndBiasError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool SpeedAndBiasError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  Eigen::Map<const okvis::SpeedAndBias> estimate(parameters[0]);
  okvis::SpeedAndBias error = measurement_ - estimate;

  Eigen::Map<Eigen::Matrix<double, 9, 1> > weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> > J0(
          jacobians[0]);
      J0 = -squareRootInformation_ * Eigen::Matrix<double, 9, 9>::Identity();
    }
  }
  if (jacobiansMinimal != nullptr) {
    if (jacobiansMinimal[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> > J0min(
          jacobiansMinimal[0]);
      J0min = -squareRootInformation_ * Eigen::Matrix<double, 9, 9>::Identity();
    }
  }

  return true;
}

}  
}  
