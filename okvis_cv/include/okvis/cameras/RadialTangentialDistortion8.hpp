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
 * @file cameras/RadialTangentialDistortion8.hpp
 * @brief Header file for the RadialTangentialDistortion class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CAMERAS_RADIALTANGENTIALDISTORTION8_HPP_
#define INCLUDE_OKVIS_CAMERAS_RADIALTANGENTIALDISTORTION8_HPP_

#include <memory>
#include <Eigen/Core>
#include "okvis/cameras/DistortionBase.hpp"

namespace okvis {

namespace cameras {

class RadialTangentialDistortion8 : public DistortionBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline RadialTangentialDistortion8();

  inline RadialTangentialDistortion8(double k1, double k2, double p1, double p2,
                                     double k3, double k4, double k5,
                                     double k6);

  inline bool setParameters(const Eigen::VectorXd & parameters) override final;

  inline bool getParameters(Eigen::VectorXd & parameters) const override final
  {
    parameters = parameters_;
    return true;
  }

  inline std::string type() const override final
  {
    return "RadialTangentialDistortion8";
  }

  inline int numDistortionIntrinsics() const override final
  {
    return NumDistortionIntrinsics;
  }

  static const int NumDistortionIntrinsics = 8;  

  static std::shared_ptr<DistortionBase> createTestObject()
  {
    return std::shared_ptr<DistortionBase>(
        new RadialTangentialDistortion8(0.6261, 0.001, -0.0002, 0.0001, 0.0001,
                                       0.9541, 0.1151, -0.0075));
  }

  static RadialTangentialDistortion8 testObject()
  {
    return RadialTangentialDistortion8(0.6261, 0.001, -0.0002, 0.0001, 0.0001,
                                      0.9541, 0.1151, -0.0075);
  }

  inline bool distort(const Eigen::Vector2d & pointUndistorted,
                      Eigen::Vector2d * pointDistorted) const override final;

  inline bool distort(const Eigen::Vector2d & pointUndistorted,
                      Eigen::Vector2d * pointDistorted,
                      Eigen::Matrix2d * pointJacobian,
                      Eigen::Matrix2Xd * parameterJacobian = nullptr) const override final;

  inline bool distortWithExternalParameters(
      const Eigen::Vector2d & pointUndistorted,
      const Eigen::VectorXd & parameters, Eigen::Vector2d * pointDistorted,
      Eigen::Matrix2d * pointJacobian = nullptr,
      Eigen::Matrix2Xd * parameterJacobian = nullptr) const override final;

  inline bool undistort(const Eigen::Vector2d & pointDistorted,
                        Eigen::Vector2d * pointUndistorted) const override final;

  inline bool undistort(const Eigen::Vector2d & pointDistorted,
                        Eigen::Vector2d * pointUndistorted,
                        Eigen::Matrix2d * pointJacobian) const override final;

 protected:
  Eigen::Matrix<double, NumDistortionIntrinsics, 1> parameters_;  

  double k1_;  
  double k2_;  
  double p1_;  
  double p2_;  
  double k3_;  
  double k4_;  
  double k5_;  
  double k6_;  
};

}  
}  

#include "implementation/RadialTangentialDistortion8.hpp"

#endif 
