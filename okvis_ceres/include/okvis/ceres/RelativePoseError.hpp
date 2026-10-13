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
 * @file RelativePoseError.hpp
 * @brief Header file for the RelativePoseError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_RELATIVEPOSEERROR_HPP_
#define INCLUDE_OKVIS_CERES_RELATIVEPOSEERROR_HPP_

#include <vector>

#include <ceres/sized_cost_function.h>

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {

namespace ceres {

class RelativePoseError : public ::ceres::SizedCostFunction<
    6 ,
    7, 
    7 >, public ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef ::ceres::SizedCostFunction<6, 7, 7> base_t;

  static const int kNumResiduals = 6;

  typedef Eigen::Matrix<double, 6, 6> information_t;

  typedef Eigen::Matrix<double, 6, 6> covariance_t;

  RelativePoseError();

  RelativePoseError(
      const Eigen::Matrix<double, 6, 6> & information,
      const kinematics::Transformation & T_AB = kinematics::Transformation::Identity());

  RelativePoseError(
      double translationVariance, double rotationVariance,
      const kinematics::Transformation & T_AB = kinematics::Transformation::Identity());

  virtual ~RelativePoseError() override = default;

  void setInformation(const information_t & information);

  const information_t& information() const {
    return information_;
  }

  const information_t& covariance() const {
    return covariance_;
  }

  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
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
    return "RelativePoseError";
  }

  const kinematics::Transformation& T_AB() const {return T_AB_;}

 protected:

  kinematics::Transformation T_AB_; 
  information_t information_; 
  information_t squareRootInformation_; 
  covariance_t covariance_; 

};

}  
}  

#endif 
