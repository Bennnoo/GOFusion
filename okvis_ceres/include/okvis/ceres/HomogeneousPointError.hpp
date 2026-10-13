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
 * @file HomogeneousPointError.hpp
 * @brief Header file for the HomogeneousPointError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTERROR_HPP_
#define INCLUDE_OKVIS_CERES_HOMOGENEOUSPOINTERROR_HPP_

#include <vector>
#include <ceres/sized_cost_function.h>
#include <Eigen/Core>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/assert_macros.hpp>

namespace okvis {

namespace ceres {

class HomogeneousPointError : public ::ceres::SizedCostFunction<
    3 , 4 >,
    public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef ::ceres::SizedCostFunction<3, 4> base_t;

  static const int kNumResiduals = 3;

  typedef Eigen::Matrix<double, 3, 3> information_t;

  typedef Eigen::Matrix<double, 3, 3> covariance_t;

  HomogeneousPointError();

  HomogeneousPointError(const Eigen::Vector4d & measurement,
                        const information_t & information);

  HomogeneousPointError(const Eigen::Vector4d & measurement, double variance);

  virtual ~HomogeneousPointError() override = default;

  void setMeasurement(const Eigen::Vector4d & measurement) {
    measurement_ = measurement;
  }

  void setInformation(const information_t & information);

  const Eigen::Vector4d& measurement() const {
    return measurement_;
  }

  const information_t& information() const {
    return information_;
  }

  const information_t& covariance() const {
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
    return int(base_t::parameter_block_sizes().size());
  }

  int parameterBlockDim(int parameterBlockId) const override final {
    return int(base_t::parameter_block_sizes().at(size_t(parameterBlockId)));
  }

  virtual std::string typeInfo() const override final {
    return "HomogeneousPointError";
  }

 protected:

  Eigen::Vector4d measurement_; 

  information_t information_; 
  information_t _squareRootInformation; 
  covariance_t covariance_; 

};

}  
}  

#endif 
