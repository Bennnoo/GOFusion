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
 * @file ParameterBlock.hpp
 * @brief Header file for the ParameterBlock class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_PARAMETERBLOCK_HPP_
#define INCLUDE_OKVIS_CERES_PARAMETERBLOCK_HPP_

#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/manifold.h>
#pragma GCC diagnostic pop

namespace okvis {

namespace ceres {

class ParameterBlock {
 public:

  ParameterBlock()
      : id_(0),
        fixed_(false),
        localParameterizationPtr_(nullptr) {
  }

  virtual ~ParameterBlock() {
  }

  virtual void setParameters(const double* parameters) = 0;

  void setId(uint64_t id) {
    id_ = id;
  }

  void setFixed(bool fixed) {
    fixed_ = fixed;
  }

  virtual double* parameters() = 0;

  virtual const double* parameters() const = 0;

  virtual int dimension() const = 0;

  virtual int minimalDimension() const = 0;

  uint64_t id() const {
    return id_;
  }

  bool fixed() const {
    return fixed_;
  }

  virtual void plus(const double* x0, const double* Delta_Chi,
                    double* x0_plus_Delta) const = 0;

  virtual void plusJacobian(const double* x0, double* jacobian) const = 0;

  virtual void minus(const double* x0, const double* x0_plus_Delta,
                     double* Delta_Chi) const = 0;

  virtual void liftJacobian(const double* x0, double* jacobian) const = 0;

  virtual void setLocalParameterizationPtr(
      const ::ceres::Manifold* localParameterizationPtr) {
    localParameterizationPtr_ = localParameterizationPtr;
  }

  virtual const ::ceres::Manifold* localParameterizationPtr() const {
    return localParameterizationPtr_;
  }

  virtual std::string typeInfo() const = 0;

 protected:

  uint64_t id_;

  bool fixed_;

  const ::ceres::Manifold* localParameterizationPtr_;
};

}  
}  

#endif 
