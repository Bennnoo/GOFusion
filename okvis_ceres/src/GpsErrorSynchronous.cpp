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
 * @file GpsErrorSynchronous.cpp
 * @brief Source File for the GpsErrorSynchronous class
 * @author Simon Boche
 */

#include <eigen3/Eigen/Core>
#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/GpsErrorSynchronous.hpp>
#include <okvis/Parameters.hpp>

namespace okvis {

namespace ceres {

GpsErrorSynchronous::GpsErrorSynchronous(
    uint64_t cameraId, const measurement_t & measurement, const covariance_t & information, const GpsParameters & gpsParameters) {
  setCameraId(cameraId);
  setMeasurement(measurement);
  setInformation(information);
  setGpsParameters(gpsParameters);
}

void GpsErrorSynchronous::setInformation(
    const covariance_t& information) {
  information_ = information;
  covariance_ = information.inverse();

  Eigen::LLT<Eigen::Matrix3d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

bool GpsErrorSynchronous::Evaluate(double const* const * parameters,
                                             double* residuals,
                                             double** jacobians) const {

  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);  
}

bool GpsErrorSynchronous::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  Eigen::Map<const Eigen::Vector3d> t_WS_W(&parameters[0][0]);
  const Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();

  Eigen::Map<const Eigen::Vector3d> t_GW_G(&parameters[1][0]);
  const Eigen::Quaterniond q_GW(parameters[1][6], parameters[1][3],parameters[1][4], parameters[1][5]);
  Eigen::Matrix3d C_GW = q_GW.toRotationMatrix();

  Eigen::Vector3d r_SA = gpsParameters_.r_SA;

  measurement_t error = measurement_ - (C_GW * (t_WS_W + C_WS*r_SA) + t_GW_G);

  measurement_t weighted_error = squareRootInformation_ * error;

  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];
  residuals[2] = weighted_error[2];

  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {

        Eigen::Matrix<double,3,6> J0_minimal;
        J0_minimal.setZero();
        J0_minimal.topLeftCorner<3,3>() = -C_GW;
        J0_minimal.topRightCorner<3,3>() = C_GW * okvis::kinematics::crossMx(C_WS*r_SA);

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J0_lift;
        PoseManifold::minusJacobian(parameters[0], J0_lift.data());

        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J0(jacobians[0]);
        J0 = squareRootInformation_ * J0_minimal * J0_lift;

        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
            J0_minimal_mapped = squareRootInformation_ * J0_minimal;  
          }
        }

    }
    if (jacobians[1] != NULL) {

        Eigen::Matrix<double,3,6> J1_minimal;
        J1_minimal.setZero();
        J1_minimal.topLeftCorner<3,3>() = -Eigen::Matrix3d::Identity();
        J1_minimal.topRightCorner<3,3>() = okvis::kinematics::crossMx(C_GW*(t_WS_W+C_WS*r_SA));

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J1_lift;
        PoseManifold::minusJacobian(parameters[1], J1_lift.data());

        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J1(jacobians[1]);
        J1 = squareRootInformation_ * J1_minimal * J1_lift;

        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor> > J1_minimal_mapped(jacobiansMinimal[1]);
            J1_minimal_mapped = squareRootInformation_ * J1_minimal;
          }
        }
    }
  }

  return true;
}

bool GpsErrorSynchronous::VerifyJacobianNumDiff(double const* const * parameters,
                                     double** jacobian) const{

  if(jacobian != NULL){

      Eigen::Map<const Eigen::Vector3d> t_GW_G(&parameters[1][0]);
      const Eigen::Quaterniond q_GW(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
      okvis::kinematics::Transformation T_GW(t_GW_G, q_GW);

      Eigen::Map<const Eigen::Vector3d> t_WS_W(&parameters[0][0]);
      const Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
      okvis::kinematics::Transformation T_WS(t_WS_W, q_WS);

      Eigen::Vector3d r_SA = gpsParameters_.r_SA;

      double dx = 1e-7;
      Eigen::Matrix<double, 6, 1> delta;
      Eigen::Matrix<double, 6, 1> xp; 
      Eigen::Matrix<double, 6, 1> xm; 

      Eigen::Matrix<double,3,1> ep; 
      Eigen::Matrix<double,3,1> em; 

      if(jacobian[0]!=NULL){

          Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > Jr(jacobian[0]);
          Eigen::Matrix<double,3,6> Jrn_minimal;
          Jrn_minimal.setZero();

          for (size_t i = 0; i < 6; ++i) {

            delta.setZero();

            delta[i] = dx;
            okvis::kinematics::Transformation Tp = T_WS;
            Tp.oplus(delta);

            ep = squareRootInformation_ * (measurement_ - (T_GW.C() * (Tp.r() + Tp.C()*r_SA) + T_GW.r()));

            delta[i] = -dx;
            okvis::kinematics::Transformation Tm = T_WS;
            Tm.oplus(delta);
            em = squareRootInformation_ * (measurement_ - (T_GW.C() * (Tm.r() + Tm.C()*r_SA) + T_GW.r()));

            Jrn_minimal.col(i) = (ep - em) / (2 * dx);
          }

          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> Jrn_lift;
          PoseManifold::minusJacobian(parameters[0], Jrn_lift.data());

          Eigen::Matrix<double, 3, 7> Jrn;
          Jrn = Jrn_minimal * Jrn_lift;

      }

      if(jacobian[1]!=NULL){

          Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > Jg(jacobian[1]);
          Eigen::Matrix<double,3,6> Jgn_minimal;
          Jgn_minimal.setZero();

          for (size_t i = 0; i < 6; ++i) {

            delta.setZero();

            delta[i] = dx;
            okvis::kinematics::Transformation Tp = T_GW;
            Tp.oplus(delta);

            ep = squareRootInformation_ * (measurement_ - (Tp.C() * (T_WS.r() + T_WS.C()*r_SA) + Tp.r()));

            delta[i] = -dx;
            okvis::kinematics::Transformation Tm = T_GW;
            Tm.oplus(delta);
            em = squareRootInformation_ * (measurement_ - (Tm.C() * (T_WS.r() + T_WS.C()*r_SA) + Tm.r()));

            Jgn_minimal.col(i) = (ep - em) / (2 * dx);
          }

          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> Jgn_lift;
          PoseManifold::minusJacobian(parameters[1], Jgn_lift.data());

          Eigen::Matrix<double, 3, 7> Jgn;
          Jgn = Jgn_minimal * Jgn_lift;

      }
      return true;

  }

  else
      return true;

} 

} 
} 
