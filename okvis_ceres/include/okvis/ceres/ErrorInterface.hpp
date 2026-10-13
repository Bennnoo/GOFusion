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
 * @file ErrorInterface.hpp
 * @brief Header file for the ErrorInterface class. A simple interface class that
          other error classes should inherit from.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_ERRORINTERFACE_HPP_
#define INCLUDE_OKVIS_CERES_ERRORINTERFACE_HPP_

#include <Eigen/Core>
#include <ceres/problem.h>
#include <ceres/cost_function.h>
#include <okvis/assert_macros.hpp>

namespace okvis {

namespace ceres {

class ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  ErrorInterface() = default;

  virtual ~ErrorInterface() = default;

  virtual int residualDim() const = 0;

  virtual int parameterBlocks() const = 0;

  virtual int parameterBlockDim(int parameterBlockId) const = 0;

  virtual bool EvaluateWithMinimalJacobians(
      double const* const * parameters, double* residuals, double** jacobians,
      double** jacobiansMinimal) const = 0;

  virtual std::string typeInfo() const = 0;
};

bool jacobiansCorrect(const ::ceres::Problem* problem, const ::ceres::ResidualBlockId id,
                      double tolerance = 1.0e-6);

}  
}  

#endif 
