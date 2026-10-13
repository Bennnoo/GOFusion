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
 * @file DistortionBase.hpp
 * @brief Header file for the DistortionBase class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CAMERAS_DISTORTIONBASE_HPP_
#define INCLUDE_OKVIS_CAMERAS_DISTORTIONBASE_HPP_

#include <Eigen/Core>
#include <okvis/cameras/CameraBase.hpp>

namespace okvis {

namespace cameras {

class DistortionBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual ~DistortionBase()
  {
  }

  virtual bool setParameters(const Eigen::VectorXd & parameters) = 0;

  virtual bool getParameters(Eigen::VectorXd & parameters) const = 0;

  virtual std::string type() const = 0;

  virtual int numDistortionIntrinsics() const = 0;

  virtual bool distort(const Eigen::Vector2d & pointUndistorted,
                       Eigen::Vector2d * pointDistorted) const = 0;

  virtual bool distort(const Eigen::Vector2d & pointUndistorted,
                       Eigen::Vector2d * pointDistorted,
                       Eigen::Matrix2d * pointJacobian,
                       Eigen::Matrix2Xd * parameterJacobian = nullptr) const = 0;

  virtual bool distortWithExternalParameters(
      const Eigen::Vector2d & pointUndistorted,
      const Eigen::VectorXd & parameters, Eigen::Vector2d * pointDistorted,
      Eigen::Matrix2d * pointJacobian = nullptr,
      Eigen::Matrix2Xd * parameterJacobian = nullptr) const = 0;

  virtual bool undistort(const Eigen::Vector2d & pointDistorted,
                         Eigen::Vector2d * pointUndistorted) const = 0;

  virtual bool undistort(const Eigen::Vector2d & pointDistorted,
                         Eigen::Vector2d * pointUndistorted,
                         Eigen::Matrix2d * pointJacobian) const = 0;

};

}  
}  

#endif 
