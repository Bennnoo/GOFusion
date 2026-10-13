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
 * @file ImuError.hpp
 * @brief Header file for the ImuError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_IMUERROR_HPP_
#define INCLUDE_OKVIS_CERES_IMUERROR_HPP_

#include <vector>
#include <mutex>
#include <atomic>

#include <ceres/sized_cost_function.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {

namespace ceres {

class ImuErrorBase : public ::ceres::SizedCostFunction<
                       15 ,
                       7 ,
                       9 ,
                       7 ,
                       9 >,
                     public ErrorInterface
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef ::ceres::SizedCostFunction<15, 7, 9, 7, 9> base_t;

  static const int kNumResiduals = 15;

  typedef Eigen::Matrix<double, 15, 15> covariance_t;

  typedef covariance_t information_t;

  typedef Eigen::Matrix<double, 15, 15> jacobian_t;

  typedef Eigen::Matrix<double, 15, 7> jacobian0_t;

  typedef Eigen::Matrix<double, 15, 9> jacobian1_t;

  virtual ~ImuErrorBase() override = default;

  void setT0(const okvis::Time &t_0) { t0_ = t_0; }

  void setT1(const okvis::Time &t_1) { t1_ = t_1; }

  okvis::Time t0() const { return t0_; }

  okvis::Time t1() const { return t1_; }

  virtual std::shared_ptr<ImuErrorBase> clone() const = 0;

  int residualDim() const override final {
    return kNumResiduals;
  }

  virtual int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final {
    return int(base_t::parameter_block_sizes().at(size_t(parameterBlockId)));
  }

  protected:

  okvis::Time t0_; 
  okvis::Time t1_; 
};

class ImuError :
    public ImuErrorBase {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  static std::atomic_bool redoPropagationAlways;

  ImuError() = default;

  virtual ~ImuError() override = default;

  ImuError(const okvis::ImuMeasurementDeque & imuMeasurements,
           const okvis::ImuParameters & imuParameters, const okvis::Time& t_0,
           const okvis::Time& t_1);

  int append(const okvis::kinematics::Transformation& T_WS,
             const okvis::SpeedAndBias & speedAndBiases,
             const okvis::ImuMeasurementDeque & imuMeasurements,
             const okvis::Time& t_1);

  static int propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                         const okvis::ImuParameters & imuParams,
                         okvis::kinematics::Transformation& T_WS,
                         okvis::SpeedAndBias & speedAndBiases,
                         const okvis::Time& t_start, const okvis::Time& t_end,
                         covariance_t* covariance = nullptr,
                         jacobian_t* jacobian = nullptr);

  static bool initPose(const okvis::ImuMeasurementDeque& imuMeasurements,
                       okvis::kinematics::Transformation& T_WS);

  int redoPreintegration(const okvis::kinematics::Transformation& T_WS,
                         const okvis::SpeedAndBias & speedAndBiases) const;

  void syncFrom(const ImuError & other);

  void setImuParameters(const okvis::ImuParameters& imuParameters);

  void setImuMeasurements(const okvis::ImuMeasurementDeque& imuMeasurements) {
    imuMeasurements_ = imuMeasurements;
  }

  virtual std::shared_ptr<ImuErrorBase> clone() const override final;

  const okvis::ImuParameters& imuParameters() const {
    return imuParameters_;
  }

  const okvis::ImuMeasurementDeque& imuMeasurements() const {
    return imuMeasurements_;
  }

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  bool EvaluateWithSigmaGradientAndHessian(double const* const* parameters,
                                            double* cost, double* gradient, double* hessian) const;

  virtual std::string typeInfo() const override final {
    return "ImuError";
  }

 protected:

  okvis::ImuParameters imuParameters_; 

  okvis::ImuMeasurementDeque imuMeasurements_; 

  mutable std::mutex preintegrationMutex_; 

  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); 
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); 
  mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); 
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); 
  mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); 

  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); 

  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); 
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); 
  mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); 

  mutable Eigen::Matrix<double,15,15> P_delta_ = Eigen::Matrix<double,15,15>::Zero();

  mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();

  mutable bool redo_ = true; 
  mutable int redoCounter_ = 0; 

  mutable information_t information_; 
  mutable information_t squareRootInformation_; 

  mutable AlignedVector<Eigen::Matrix<double,15,15>> dPdsigma_;

};

class PseudoImuError : public ImuErrorBase {
  public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  PseudoImuError() = default;

  PseudoImuError(const okvis::Time& t_0, const okvis::Time& t_1);

  virtual ~PseudoImuError() override = default;

  static int propagation(okvis::kinematics::Transformation& T_WS,
                         okvis::SpeedAndBias & speedAndBiases,
                         const okvis::Time& t_start, const okvis::Time& t_end);

  void syncFrom(const PseudoImuError & other);

  virtual std::shared_ptr<ImuErrorBase> clone() const override final;

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  virtual std::string typeInfo() const override final {
    return "PseudoImuError";
  }

  protected:

};

}  
}  

#endif 
