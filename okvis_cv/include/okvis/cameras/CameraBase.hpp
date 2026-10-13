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
 * @file cameras/CameraBase.hpp
 * @brief Header file for the CameraBase class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CAMERAS_CAMERABASE_HPP_
#define INCLUDE_OKVIS_CAMERAS_CAMERABASE_HPP_

#include <vector>
#include <stdint.h>
#include <Eigen/Core>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp> 
#pragma GCC diagnostic pop

namespace okvis {

namespace cameras {

enum class ProjectionStatus
{
  Successful,
  OutsideImage,
  Masked,
  Behind,
  Invalid
};

class CameraBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline CameraBase()
      : imageWidth_(0),
        imageHeight_(0),
        id_(0)
  {
  }

  inline CameraBase(int imageWidth, int imageHeight, uint64_t id = 0)
        : imageWidth_(imageWidth),
          imageHeight_(imageHeight),
          id_(id)
  {
  }

  inline virtual ~CameraBase()
  {
  }

  inline bool setMask(const cv::Mat & mask);

  inline bool hasMask() const;

  inline bool removeMask();

  inline const cv::Mat & mask() const;

  inline void setId(uint64_t id)
  {
    id_ = id;
  }

  inline uint64_t id() const
  {
    return id_;
  }

  inline uint32_t imageWidth() const
  {
    return imageWidth_;
  }

  inline uint32_t imageHeight() const
  {
    return imageHeight_;
  }

  virtual void getIntrinsics(Eigen::VectorXd & intrinsics) const = 0;

  virtual bool setIntrinsics(const Eigen::VectorXd & intrinsics) = 0;


  virtual void getRectifyMap(cv::Mat & map1, cv::Mat & map2) const = 0;

  virtual ProjectionStatus project(const Eigen::Vector3d & point,
                                   Eigen::Vector2d * imagePoint) const = 0;

  virtual ProjectionStatus project(
      const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint,
      Eigen::Matrix<double, 2, 3> * pointJacobian,
      Eigen::Matrix2Xd * intrinsicsJacobian = nullptr) const = 0;

  virtual ProjectionStatus projectWithExternalParameters(
      const Eigen::Vector3d & point, const Eigen::VectorXd & parameters,
      Eigen::Vector2d * imagePoint, Eigen::Matrix<double, 2, 3> * pointJacobian = nullptr,
      Eigen::Matrix2Xd * intrinsicsJacobian = nullptr) const = 0;

  virtual void projectBatch(const Eigen::Matrix3Xd & points,
                            Eigen::Matrix2Xd * imagePoints,
                            std::vector<ProjectionStatus> * stati) const = 0;

  virtual ProjectionStatus projectHomogeneous(
      const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint) const = 0;

  virtual ProjectionStatus projectHomogeneous(
      const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint,
      Eigen::Matrix<double, 2, 4> * pointJacobian,
      Eigen::Matrix2Xd * intrinsicsJacobian = nullptr) const = 0;

  virtual ProjectionStatus projectHomogeneousWithExternalParameters(
      const Eigen::Vector4d & point, const Eigen::VectorXd & parameters,
      Eigen::Vector2d * imagePoint,
      Eigen::Matrix<double, 2, 4> * pointJacobian = nullptr,
      Eigen::Matrix2Xd * intrinsicsJacobian = nullptr) const = 0;

  virtual void projectHomogeneousBatch(
      const Eigen::Matrix4Xd & points, Eigen::Matrix2Xd * imagePoints,
      std::vector<ProjectionStatus> * stati) const = 0;

  virtual bool backProject(const Eigen::Vector2d & imagePoint,
                           Eigen::Vector3d * direction) const = 0;

  virtual bool backProject(
      const Eigen::Vector2d & imagePoint, Eigen::Vector3d * direction,
      Eigen::Matrix<double, 3, 2> * pointJacobian) const = 0;

  virtual bool backProjectBatch(const Eigen::Matrix2Xd & imagePoints,
                                Eigen::Matrix3Xd * directions,
                                std::vector<bool> * success) const = 0;

  virtual bool backProjectHomogeneous(const Eigen::Vector2d & imagePoint,
                                      Eigen::Vector4d * direction) const = 0;

  virtual bool backProjectHomogeneous(
      const Eigen::Vector2d & imagePoint, Eigen::Vector4d * direction,
      Eigen::Matrix<double, 4, 2> * pointJacobian) const = 0;

  virtual bool backProjectHomogeneousBatch(
      const Eigen::Matrix2Xd & imagePoints, Eigen::Matrix4Xd * directions,
      std::vector<bool> * success) const = 0;

  virtual Eigen::Vector2d createRandomImagePoint() const;

  virtual Eigen::Vector3d createRandomVisiblePoint(double minDist = 0.0,
                                                   double maxDist = 10.0) const;

  virtual Eigen::Vector4d createRandomVisibleHomogeneousPoint(double minDist =
                                                                  0.0,
                                                              double maxDist =
                                                                  10.0) const;

  virtual int noIntrinsicsParameters() const = 0;

  virtual std::string type() const = 0;

  virtual const std::string distortionType() const = 0;

  virtual bool initialiseCameraAwarenessMaps() = 0;

  virtual bool getCameraAwarenessMaps(cv::Mat& rays, cv::Mat& imageJacobians) const = 0;

 protected:

  inline bool isMasked(const Eigen::Vector2d& imagePoint) const;

  inline bool isInImage(const Eigen::Vector2d& imagePoint) const;

  cv::Mat mask_;  

  int imageWidth_;  
  int imageHeight_;  

  uint64_t id_;  

};

}  
}  

#include "implementation/CameraBase.hpp"

#endif 
