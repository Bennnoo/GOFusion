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
 * @file ParameterBlockSized.hpp
 * @brief Header file for the ParameterBlockSized class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_PARAMETERBLOCKSIZED_HPP_
#define INCLUDE_OKVIS_CERES_PARAMETERBLOCKSIZED_HPP_

#include <okvis/ceres/ParameterBlock.hpp>
#include <okvis/assert_macros.hpp>
#include <Eigen/Core>

namespace okvis {

namespace ceres {

template<int Dim, int MinDim, class T>
class ParameterBlockSized : public okvis::ceres::ParameterBlock {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  static const int Dimension = Dim;

  static const int MinimalDimension = MinDim;

  typedef T parameter_t;

  ParameterBlockSized() = default;

  virtual ~ParameterBlockSized() override = default;

  void setEstimate(const parameter_t& estimate) {
    estimate_ = estimate;
  }

  const parameter_t& estimate() const {
    return estimate_;
  }

  virtual int dimension() const override final {
    return Dimension;
  }

  virtual int minimalDimension() const override final {
    return MinimalDimension;
  }

 protected:

  parameter_t estimate_;
};

}  
}  

#endif 
