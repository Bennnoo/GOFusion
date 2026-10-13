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
 * @file NoDistortion.hpp
 * @brief Header file for the NoDistortion class.
 * @author Stefan Leutenegger 
 */

#ifndef INCLUDE_OKVIS_CAMERAS_NODISTORTION_HPP_
#define INCLUDE_OKVIS_CAMERAS_NODISTORTION_HPP_

#include <memory>
#include <Eigen/Core>
#include "okvis/cameras/DistortionBase.hpp"

namespace okvis {

namespace cameras {

class NoDistortion : public DistortionBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual inline ~NoDistortion() override = default;

  bool setParameters(const Eigen::VectorXd & parameters) override final
  {
    (void)parameters;
    return true;
  }

  bool getParameters(Eigen::VectorXd & parameters) const override final
  {
    parameters.resize(0);
    return true;
  }

  std::string type() const override final
  {
    return "NoDistortion";
  }

  int numDistortionIntrinsics() const override final
  {
    return 0;
  }

  static const int NumDistortionIntrinsics = 0; 

  static std::shared_ptr<DistortionBase> createTestObject() {
    return std::shared_ptr<DistortionBase>(new NoDistortion());
  }

  static NoDistortion testObject() {
    return NoDistortion();
  }

  bool distort(const Eigen::Vector2d & pointUndistorted,
                       Eigen::Vector2d * pointDistorted) const override final
  {
    *pointDistorted = pointUndistorted;
    return true;
  }

  bool distort(const Eigen::Vector2d & pointUndistorted,
                       Eigen::Vector2d * pointDistorted,
                       Eigen::Matrix2d * pointJacobian,
                       Eigen::Matrix2Xd * parameterJacobian = nullptr) const override final
  {
    *pointDistorted = pointUndistorted;
    *pointJacobian = Eigen::Matrix2d::Identity();
    if (parameterJacobian) {
      parameterJacobian->resize(2, 0);
    }
    return true;
  }

  bool distortWithExternalParameters(
      const Eigen::Vector2d & pointUndistorted,
      const Eigen::VectorXd & parameters, Eigen::Vector2d * pointDistorted,
      Eigen::Matrix2d * pointJacobian = nullptr,
      Eigen::Matrix2Xd * parameterJacobian = nullptr) const override final
  {
    (void)parameters;
    *pointDistorted = pointUndistorted;
    if (pointJacobian) {
      *pointJacobian = Eigen::Matrix2d::Identity();
    }
    if (parameterJacobian) {
      parameterJacobian->resize(2, 0);
    }
    return true;
  }

  bool undistort(const Eigen::Vector2d & pointDistorted,
                 Eigen::Vector2d * pointUndistorted) const override final
  {
    *pointUndistorted = pointDistorted;
    return true;
  }

  bool undistort(const Eigen::Vector2d & pointDistorted,
                         Eigen::Vector2d * pointUndistorted,
                         Eigen::Matrix2d * pointJacobian) const override final
  {
    *pointUndistorted = pointDistorted;
    *pointJacobian = Eigen::Matrix2d::Identity();
    return true;
  }

};

}  
}  

#endif 
