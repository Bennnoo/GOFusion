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
 * @file DepthError.hpp
 * @brief Header file for the PoseError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_DEPTHERROR_HPP_
#define INCLUDE_OKVIS_CERES_DEPTHERROR_HPP_

#include <ceres/sized_cost_function.h>

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

namespace okvis {

namespace ceres {

template<bool ONESIDED>
class DepthErrorT : public ::ceres::SizedCostFunction<
                     1 ,
                     7 ,
                     4 ,
                     7 >, public ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef ::ceres::SizedCostFunction<1, 7, 4, 7> base_t;

  static const bool onesided = ONESIDED; 

  static const int kNumResiduals = 1;

  DepthErrorT() {}

  DepthErrorT(double measurement, double stdev)
  {
    measurement_ = measurement;
    information_ = 1.0/(stdev*stdev);
    squareRootInformation_ = sqrt(information_);
  }

  virtual ~DepthErrorT() override = default;

  void setMeasurement(double measurement) { measurement_ = measurement; }

  void setStdev(double stdev)
  {
    information_ = 1.0/(stdev*stdev);
    squareRootInformation_ = sqrt(information_);
  }

  double measurement() const { return measurement_; }

  double information() const { return information_; }

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final {
    return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
  }

  virtual bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                            double *residuals,
                                            double **jacobians,
                                            double **jacobiansMinimal) const override final
  {

    Eigen::Map<const Eigen::Vector3d> t_WS_W(&parameters[0][0]);
    Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    q_WS.normalize();

    Eigen::Map<const Eigen::Vector4d> hp_W(&parameters[1][0]);

    Eigen::Map<const Eigen::Vector3d> t_SC_S(&parameters[2][0]);
    Eigen::Quaterniond q_SC(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
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
    double p_z = 0.0;

    bool ignore = false;
    if (fabs(hp_C[3]) < 1.0e-16) {
      ignore = true;
    } else if (onesided) {
      p_z = hp_C[2] / hp_C[3];
      if (p_z > measurement_) {
        ignore = true;
      }
    }

    const double error = measurement_ - p_z;

    if (ignore) {
      residuals[0] = 0.0;
    } else {
      residuals[0] = squareRootInformation_ * error;
    }

    if (jacobians != nullptr) {
      Eigen::Matrix<double, 1, 4> Jh_weighted(0.0, 0.0, squareRootInformation_ / hp_C[3], 0.0);
      if (jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J0(jacobians[0]);
        if (ignore) {
          J0.setZero();
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[0] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J0_minimal_mapped(
                jacobiansMinimal[0]);
              J0_minimal_mapped.setZero();
            }
          }
        } else {
          Eigen::Vector3d p = hp_W.head<3>() - t_WS_W * hp_W[3];
          Eigen::Matrix<double, 4, 6> J;
          J.setZero();
          J.topLeftCorner<3, 3>() = C_SW * hp_W[3];
          J.topRightCorner<3, 3>() = -C_SW * okvis::kinematics::crossMx(p);

          Eigen::Matrix<double, 1, 6, Eigen::RowMajor> J0_minimal;
          J0_minimal = Jh_weighted * T_CS * J;

          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
          PoseManifold::minusJacobian(parameters[0], J_lift.data());

          J0 = J0_minimal * J_lift;

          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[0] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J0_minimal_mapped(
                jacobiansMinimal[0]);
              J0_minimal_mapped = J0_minimal;
            }
          }
        }
      }
      if (jacobians[1] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor> > J1(
          jacobians[1]); 
        if (ignore) {
          J1.setZero();
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[1] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J1_minimal_mapped(
                jacobiansMinimal[1]);
              J1_minimal_mapped.setZero();
            }
          }
        } else {
          Eigen::Matrix4d T_CW = (T_CS * T_SW);
          J1 = -Jh_weighted * T_CW;

          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[1] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J1_minimal_mapped(
                jacobiansMinimal[1]);
              Eigen::Matrix<double, 4, 3> S;
              S.setZero();
              S.topLeftCorner<3, 3>().setIdentity();
              J1_minimal_mapped = J1 * S; 
            }
          }
        }
      }
      if (jacobians[2] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J2(jacobians[2]);
        if (ignore) {
          J2.setZero();
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[2] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J2_minimal_mapped(
                jacobiansMinimal[2]);
              J2_minimal_mapped.setZero();
            }
          }
        } else {
          Eigen::Vector3d p = hp_S.head<3>() - t_SC_S * hp_S[3];
          Eigen::Matrix<double, 4, 6> J;
          J.setZero();
          J.topLeftCorner<3, 3>() = C_CS * hp_S[3];
          J.topRightCorner<3, 3>() = -C_CS * okvis::kinematics::crossMx(p);

          Eigen::Matrix<double, 1, 6, Eigen::RowMajor> J2_minimal;
          J2_minimal = Jh_weighted * J;

          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
          PoseManifold::minusJacobian(parameters[2], J_lift.data());

          J2 = J2_minimal * J_lift;

          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[2] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J2_minimal_mapped(
                jacobiansMinimal[2]);
              J2_minimal_mapped = J2_minimal;
            }
          }
        }
      }
    }

    return true;

  }

  int residualDim() const override final {
    return kNumResiduals;
  }

  int parameterBlocks() const override final {
    return int(base_t::parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final{
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }

  virtual std::string typeInfo() const override final {
    return "DepthError";
  }

 protected:

  double measurement_; 

  double information_; 
  double squareRootInformation_; 

};

typedef DepthErrorT<true> OneSidedDepthError;

typedef DepthErrorT<false> DepthError;

}  
}  

#endif 
