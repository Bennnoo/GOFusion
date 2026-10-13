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
 * @file SpeedAndBiasError.hpp
 * @brief Header file for the SpeedAndBiasError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_SPEEDANDBIASERROR_HPP_
#define INCLUDE_OKVIS_CERES_SPEEDANDBIASERROR_HPP_

#include <vector>
#include <Eigen/Core>

#include <ceres/sized_cost_function.h>

#include <okvis/assert_macros.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {

namespace ceres {

class SpeedAndBiasError : public ::ceres::SizedCostFunction<
    9 ,
    9 >,
    public ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef ::ceres::SizedCostFunction<9, 9> base_t;

  static const int kNumResiduals = 9;

  typedef Eigen::Matrix<double, 9, 9> information_t;

  typedef Eigen::Matrix<double, 9, 9> covariance_t;

  SpeedAndBiasError();

  SpeedAndBiasError(const okvis::SpeedAndBias & measurement,
                    const information_t & information);

  SpeedAndBiasError(const okvis::SpeedAndBias& measurement,
                    double speedVariance, double gyrBiasVariance,
                    double accBiasVariance);

  virtual ~SpeedAndBiasError() override = default;

  void setMeasurement(const okvis::SpeedAndBias & measurement) {
    measurement_ = measurement;
  }

  void setInformation(const information_t & information);

  const okvis::SpeedAndBias& measurement() const {
    return measurement_;
  }

  const information_t& information() const {
    return information_;
  }

  const covariance_t& covariance() const {
    return covariance_;
  }

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const override final;

  int residualDim() const override final {
    return kNumResiduals;
  }

  int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final {
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }

  virtual std::string typeInfo() const override final {
    return "SpeedAndBiasError";
  }

 protected:

  okvis::SpeedAndBias measurement_; 

  information_t information_; 
  information_t squareRootInformation_; 
  covariance_t covariance_; 

};

}  
}  

#endif 
