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
 * @file cameras/PinholeCamera.hpp
 * @brief Header file for the PinholeCamera class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_CAMERAS_PINHOLECAMERA_HPP_
#define INCLUDE_OKVIS_CAMERAS_PINHOLECAMERA_HPP_

#include <vector>
#include <memory>
#include <stdint.h>
#include <Eigen/Core>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp> 
#pragma GCC diagnostic pop
#include "okvis/cameras/CameraBase.hpp"
#include "okvis/cameras/DistortionBase.hpp"
#include "okvis/cameras/NoDistortion.hpp"

namespace okvis {

namespace cameras {

template<class DISTORTION_T>
class PinholeCamera; 

class PinholeCameraBase : public CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline PinholeCameraBase(int imageWidth, int imageHeight, uint64_t id = 0)
        : CameraBase(imageWidth, imageHeight, id)
  {
  }

  virtual ~PinholeCameraBase() override = default;

  virtual bool initialiseUndistortMaps() = 0;

  virtual bool initialiseUndistortMaps(int undistortedImageWidth, int undistortedImageHeight, 
      double undistortedFocalLengthU, double undistortedFocalLengthV, 
      double undistortedImageCenterU, double undistortedImageCenterV) = 0;

  virtual bool undistortImage(const cv::Mat & srcImg, cv::Mat & destImg) const = 0;

  virtual PinholeCamera<NoDistortion> undistortedPinholeCamera() const = 0;

  virtual double focalLengthU() const = 0;

  virtual double focalLengthV() const = 0;

  virtual double imageCenterU() const = 0;

  virtual double imageCenterV() const = 0;
};

template<class DISTORTION_T>
class PinholeCamera : public PinholeCameraBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef DISTORTION_T distortion_t; 

  PinholeCamera(int imageWidth, int imageHeight, double focalLengthU,
                double focalLengthV, double imageCenterU, double imageCenterV,
                const distortion_t & distortion, uint64_t id=std::numeric_limits<uint64_t>::max());

  virtual ~PinholeCamera() override = default;

  static const int NumProjectionIntrinsics = 4;  
  static const int NumIntrinsics = NumProjectionIntrinsics
      + distortion_t::NumDistortionIntrinsics; 

  virtual double focalLengthU() const override final
  {
    return fu_;
  }

  virtual double focalLengthV() const override final
  {
    return fv_;
  }

  virtual double imageCenterU() const override final
  {
    return cu_;
  }

  virtual double imageCenterV() const override final
  {
    return cv_;
  }

  inline void getIntrinsics(Eigen::VectorXd & intrinsics) const override final;

  inline bool setIntrinsics(const Eigen::VectorXd & intrinsics) override final;


  virtual void getRectifyMap(cv::Mat & map1, cv::Mat & map2) const;

  inline int noIntrinsicsParameters() const override final
  {
    return NumIntrinsics;
  }

  virtual bool initialiseUndistortMaps() override final;

  virtual bool initialiseCameraAwarenessMaps() override final;

  virtual bool getCameraAwarenessMaps(cv::Mat& rays, cv::Mat& imageJacobians) const override final;

  virtual bool initialiseUndistortMaps(int undistortedImageWidth, int undistortedImageHeight, 
      double undistortedFocalLengthU, double undistortedFocalLengthV, 
      double undistortedImageCenterU, double undistortedImageCenterV) override final;

  virtual PinholeCamera<NoDistortion> undistortedPinholeCamera() const override final;

  virtual bool undistortImage(const cv::Mat & srcImg, cv::Mat & destImg) const override final;

  inline ProjectionStatus project(
      const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint) const override final;

  inline ProjectionStatus project(
      const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint,
      Eigen::Matrix<double, 2, 3> * pointJacobian,
      Eigen::Matrix2Xd * intrinsicsJacobian = nullptr) const override final;

  inline ProjectionStatus projectWithExternalParameters(
      const Eigen::Vector3d & point, const Eigen::VectorXd & parameters,
      Eigen::Vector2d * imagePoint, Eigen::Matrix<double, 2, 3> * pointJacobian,
      Eigen::Matrix2Xd * intrinsicsJacobian = nullptr) const override final;

  inline void projectBatch(
      const Eigen::Matrix3Xd & points, Eigen::Matrix2Xd * imagePoints,
      std::vector<ProjectionStatus> * stati) const override final;

  inline ProjectionStatus projectHomogeneous(
      const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint) const override final;

  inline ProjectionStatus projectHomogeneous(
      const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint,
      Eigen::Matrix<double, 2, 4> * pointJacobian,
      Eigen::Matrix2Xd * intrinsicsJacobian = nullptr) const override final;

  inline ProjectionStatus projectHomogeneousWithExternalParameters(
      const Eigen::Vector4d & point, const Eigen::VectorXd & parameters,
      Eigen::Vector2d * imagePoint,
      Eigen::Matrix<double, 2, 4> * pointJacobian = nullptr,
      Eigen::Matrix2Xd * intrinsicsJacobian = nullptr) const override final;

  inline void projectHomogeneousBatch(
      const Eigen::Matrix4Xd & points, Eigen::Matrix2Xd * imagePoints,
      std::vector<ProjectionStatus> * stati) const override final;

  inline bool backProject(const Eigen::Vector2d & imagePoint,
                          Eigen::Vector3d * direction) const override final;

  inline bool backProject(const Eigen::Vector2d & imagePoint,
                          Eigen::Vector3d * direction,
                          Eigen::Matrix<double, 3, 2> * pointJacobian) const override final;

  inline bool backProjectBatch(const Eigen::Matrix2Xd & imagePoints,
                               Eigen::Matrix3Xd * directions,
                               std::vector<bool> * success) const override final;

  inline bool backProjectHomogeneous(const Eigen::Vector2d & imagePoint,
                                     Eigen::Vector4d * direction) const override final;

  inline bool backProjectHomogeneous(
      const Eigen::Vector2d & imagePoint, Eigen::Vector4d * direction,
      Eigen::Matrix<double, 4, 2> * pointJacobian) const override final;

  inline bool backProjectHomogeneousBatch(const Eigen::Matrix2Xd & imagePoints,
                                          Eigen::Matrix4Xd * directions,
                                          std::vector<bool> * success) const override final;

  static std::shared_ptr<CameraBase> createTestObject()
  {
    return std::shared_ptr<CameraBase>(new PinholeCamera(752, 480, 350, 360, 378, 238,
                         distortion_t::testObject()));
  }

  static PinholeCamera testObject()
  {
    return PinholeCamera(752, 480, 350, 360, 378, 238,
                          distortion_t::testObject());
  }

  std::string type() const override final
  {
    return "PinholeCamera<" + distortion_.type() + ">";
  }

  const std::string distortionType() const override final
  {
    return distortion_.type();
  }

 protected:

  PinholeCamera() = delete;

  distortion_t distortion_;  

  Eigen::Matrix<double, NumIntrinsics, 1> intrinsics_;  
  double fu_;  
  double fv_;  
  double cu_;  
  double cv_;  
  double one_over_fu_;  
  double one_over_fv_;  
  double fu_over_fv_;  

  cv::Mat map_x_fast_; 
  cv::Mat map_y_fast_; 

  int undistortedImageWidth_ = 0;  
  int undistortedImageHeight_ = 0;  
  double undistortedFocalLengthU_ = 0.0;  
  double undistortedFocalLengthV_ = 0.0;  
  double undistortedImageCenterU_ = 0.0;  
  double undistortedImageCenterV_ = 0.0;  

  cv::Mat rays_;  
  cv::Mat imageJacobians_;  

};

}  
}  

#include "implementation/PinholeCamera.hpp"

#endif 
