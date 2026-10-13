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
 * @file ceres/GpsErrorAsynchronous.hpp
 * @brief Header file for the asynchronous GpsError class.
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_CERES_GPSERRORASYNCHRONOUS_HPP_
#define INCLUDE_OKVIS_CERES_GPSERRORASYNCHRONOUS_HPP_

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <ceres/sized_cost_function.h>
#include <ceres/covariance.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {
namespace ceres {

class GpsErrorAsynchronous :
    public ::ceres::SizedCostFunction<3 ,
        7 ,
        9 ,
        7 >,
    public ErrorInterface {

 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static std::atomic_bool redoPropagationAlways;
  static std::atomic_bool useImuCovariance;

  typedef ::ceres::SizedCostFunction<3, 7, 9, 7> base_t;

  static const int kNumResiduals = 3;

  typedef Eigen::Matrix<double, 3, 3> covariance_t;

  typedef Eigen::Vector3d measurement_t;

  typedef covariance_t information_t;

  typedef Eigen::Matrix<double, 3, 7> jacobian0_t;

  typedef Eigen::Matrix<double, 3, 9> jacobian1_t;

  typedef Eigen::Matrix<double, 3, 7> jacobian2_t;

  GpsErrorAsynchronous();

  GpsErrorAsynchronous(const measurement_t & measurement, const covariance_t & information,
                       const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                       const okvis::Time& tk, const okvis::Time& tg,
                       const GpsParameters& gpsParameters);

  GpsErrorAsynchronous(const measurement_t & measurement, const double sigma_x, const double sigma_y, const double sigma_z,
                       const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                       const okvis::Time& tk, const okvis::Time& tg,
                       const GpsParameters& gpsParameters);

  virtual ~GpsErrorAsynchronous()
  {
  }

  virtual void setGpsParameters(const GpsParameters & gpsParameters){
      gpsParameters_ = gpsParameters;
  }

  virtual void setMeasurement(const measurement_t& measurement)
  {
    measurement_ = measurement;
  }

  virtual void setInformation(const covariance_t& information);

  void setTg(const okvis::Time& tg)
  {
    tg_ = tg;
  }

  void setTk(const okvis::Time& tk)
  {
    tk_ = tk;
  }

  void setImuParameters(const okvis::ImuParameters& imuParameters){
      imuParameters_ = imuParameters;
  }

  void setImuMeasurements(const okvis::ImuMeasurementDeque& imuMeasurements) {
    imuMeasurements_ = imuMeasurements;
  }

  virtual const measurement_t& measurement() const
  {
    return measurement_;
  }

  virtual const covariance_t& information() const
  {
    return gpsInformation_;
  }

  virtual const covariance_t& covariance() const
  {
    return gpsCovariance_;
  }

  okvis::Time tg() const
  {
    return tg_;
  }

  okvis::Time tk() const
  {
    return tk_;
  }

  const okvis::ImuParameters& imuParameters() const {
    return imuParameters_;
  }

  const okvis::ImuMeasurementDeque& imuMeasurements() const {
    return imuMeasurements_;
  }

  const okvis::GpsParameters& gpsParameters() const {
    return gpsParameters_;
  }

  const Eigen::Vector3d error() const {
      return error_;
  }

  bool applyPreInt(const okvis::kinematics::Transformation T_WS_in, const SpeedAndBias sb_in, okvis::kinematics::Transformation& T_WS_prop){

    const Eigen::Matrix3d C_WS_tk = T_WS_in.C();
    const Eigen::Vector3d r_WS_tk = T_WS_in.r();

    const double Delta_t = (tg_ - tk_).toSec();
    Eigen::Matrix<double, 6, 1> Delta_b;

    {
      Delta_b = sb_in.tail<6>()
                - speedAndBiases_ref_.tail<6>();
      redo_ = redo_ || (Delta_b.head<3>().norm() > 0.0003);

    }

    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    T_WS_prop.set(r_WS_tk + sb_in.head<3>()*Delta_t - 0.5*g_W*Delta_t*Delta_t
                + C_WS_tk * (acc_doubleintegral_ + dp_db_g_ * Delta_b.head<3>() - C_doubleintegral_ * Delta_b.tail<3>()),
                T_WS_in.q()*Delta_q_*okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b.head<3>()));

    return true;

  }

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  int redoPreintegration(const okvis::kinematics::Transformation& T_WS,
                         const okvis::SpeedAndBias & speedAndBiases) const;

  int residualDim() const
  {
    return kNumResiduals;
  }

  int parameterBlocks() const
  {
    return parameter_block_sizes().size();
  }

  int parameterBlockDim(int parameterBlockId) const
  {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  virtual std::string typeInfo() const
  {
    return "GpsErrorAsynchronous";
  }

 protected:

  measurement_t measurement_; 

  mutable Eigen::Vector3d error_; 

  GpsParameters gpsParameters_; 

  covariance_t gpsInformation_; 

  covariance_t gpsCovariance_; 

  okvis::ImuParameters imuParameters_; 

  okvis::ImuMeasurementDeque imuMeasurements_; 

  okvis::Time tk_; 
  okvis::Time tg_; 

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

  mutable AlignedVector<Eigen::Matrix<double,15,15>> dPdsigma_;

  mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();

  mutable bool redo_ = true; 
  mutable int redoCounter_ = 0; 

  mutable information_t squareRootInformation_; 

};

}  
}  

#endif 
