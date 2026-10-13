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
 * @file implementation/PinholeCamera.hpp
 * @brief Header implementation file for the PinholeCamera class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#pragma once

#include <Eigen/Geometry>

#include <okvis/cameras/PinholeCamera.hpp>

namespace okvis {

namespace cameras {

template<class DISTORTION_T>
PinholeCamera<DISTORTION_T>::PinholeCamera(int imageWidth,
                                           int imageHeight,
                                           double focalLengthU,
                                           double focalLengthV,
                                           double imageCenterU,
                                           double imageCenterV,
                                           const distortion_t & distortion,
                                           uint64_t id)
    : PinholeCameraBase(imageWidth, imageHeight, id),
    distortion_(distortion),
    fu_(focalLengthU),
    fv_(focalLengthV),
    cu_(imageCenterU),
    cv_(imageCenterV)
{
  intrinsics_[0] = fu_;  
  intrinsics_[1] = fv_;  
  intrinsics_[2] = cu_;  
  intrinsics_[3] = cv_;  
  one_over_fu_ = 1.0 / fu_;  
  one_over_fv_ = 1.0 / fv_;  
  fu_over_fv_ = fu_ / fv_;  
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::setIntrinsics(
    const Eigen::VectorXd & intrinsics)
{
  if (intrinsics.cols() != NumIntrinsics) {
    return false;
  }
  intrinsics_ = intrinsics;
  fu_ = intrinsics[0];  
  fv_ = intrinsics[1];  
  cu_ = intrinsics[2];  
  cv_ = intrinsics[3];  
  distortion_.setParameters(
      intrinsics.tail<distortion_t::NumDistortionIntrinsics>());
  one_over_fu_ = 1.0 / fu_;  
  one_over_fv_ = 1.0 / fv_;  
  fu_over_fv_ = fu_ / fv_;  
  return true;
}

template<class DISTORTION_T>
void PinholeCamera<DISTORTION_T>::getRectifyMap(cv::Mat & map1, cv::Mat & map2) const {
  map1 = map_x_fast_.clone();
  map2 = map_y_fast_.clone();
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::initialiseUndistortMaps() {
  const double f = (fu_ + fv_) * 0.5;
  return initialiseUndistortMaps(imageWidth(), imageHeight(), f, f, 
      double(imageWidth())*0.5+0.5, double(imageHeight())*0.5+0.5);
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::initialiseUndistortMaps(
    int undistortedImageWidth, int undistortedImageHeight, 
    double undistortedFocalLengthU, double undistortedFocalLengthV, 
    double undistortedImageCenterU, double undistortedImageCenterV) {

  undistortedImageWidth_ = undistortedImageWidth;
  undistortedImageHeight_ = undistortedImageHeight;
  undistortedFocalLengthU_ = undistortedFocalLengthU;
  undistortedFocalLengthV_ = undistortedFocalLengthV;
  undistortedImageCenterU_ = undistortedImageCenterU;
  undistortedImageCenterV_ = undistortedImageCenterV;

  Eigen::Matrix2d undistortedK_inv, actualK;
  undistortedK_inv << 1.0/undistortedFocalLengthU, 0.0, 0.0, 1.0/undistortedFocalLengthV;
  actualK << fu_, 0.0, 0.0, fv_;
  Eigen::Vector2d actualCenter(cu_, cv_);
  Eigen::Vector2d undistortedCenter(undistortedImageCenterU, undistortedImageCenterV);

  cv::Mat map_x(undistortedImageHeight, undistortedImageWidth, CV_32F);
  cv::Mat map_y(undistortedImageHeight, undistortedImageWidth, CV_32F);
  Eigen::Vector2d pixel, imgPlane, projectedPoint, distortedPoint, mappedPixel;
  Eigen::Vector3d ray, rayTransformed;
  Eigen::Vector4d hrayTransformed;
  const double rayLength = 0.25;

  for (int y = 0; y < undistortedImageHeight; y++) {

    pixel(1) = y;

    float *pmap_x = map_x.ptr<float>(y); 
    float *pmap_y = map_y.ptr<float>(y);

    for (int x = 0; x < undistortedImageWidth; x++) {

      pixel(0) = x;

      imgPlane = undistortedK_inv * (pixel - undistortedCenter);

      ray = rayLength * imgPlane.homogeneous();

      hrayTransformed =  ray.homogeneous();
      rayTransformed = hrayTransformed.hnormalized();

      projectedPoint = rayTransformed.hnormalized();

      distortion_.distort(projectedPoint,&distortedPoint);

      mappedPixel = (actualK * distortedPoint) + actualCenter;

      pmap_x[x] = mappedPixel(0); 
      pmap_y[x] = mappedPixel(1);

    }
  }

  cv::convertMaps(map_x, map_y, map_x_fast_, map_y_fast_, CV_16SC2);

  return true;
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::initialiseCameraAwarenessMaps() {
  rays_ = cv::Mat(imageHeight_, imageWidth_, CV_32FC3);
  imageJacobians_ = cv::Mat(imageHeight_, imageWidth_, CV_32FC(6));
  for (int v=0; v<imageHeight_; ++v) {
    for (int u=0; u<imageWidth_; ++u) {
      Eigen::Vector3d ray;
      Eigen::Matrix<double, 2, 3> jacobian;
      if(backProject(Eigen::Vector2d(u,v), &ray)) {
        ray.normalize();
      } else {
        ray.setZero();
      }
      rays_.at<cv::Vec3f>(v,u) = cv::Vec3f(ray[0],ray[1],ray[2]);
      Eigen::Vector2d pt;
      if(project(ray, &pt, &jacobian)
         ==cameras::ProjectionStatus::Successful) {
        cv::Vec6f j;
        j[0]=jacobian(0,0);
        j[1]=jacobian(0,1);
        j[2]=jacobian(0,2);
        j[3]=jacobian(1,0);
        j[4]=jacobian(1,1);
        j[5]=jacobian(1,2);
        imageJacobians_.at<cv::Vec6f>(v,u) = j;
      }
    }
  }
  return true;
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::getCameraAwarenessMaps(
    cv::Mat& rays, cv::Mat& imageJacobians) const {
  rays = rays_;
  imageJacobians = imageJacobians_;
  if(rays_.empty() || imageJacobians_.empty()) {
    return false;
  }
  return true;
}

template<class DISTORTION_T>
PinholeCamera<NoDistortion> PinholeCamera<DISTORTION_T>::undistortedPinholeCamera()
    const {
  assert(map_x_fast_.cols !=0);
  return PinholeCamera<NoDistortion>(undistortedImageWidth_, undistortedImageHeight_,
      undistortedFocalLengthU_, undistortedFocalLengthV_,
      undistortedImageCenterU_, undistortedImageCenterV_,
      NoDistortion());
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::undistortImage(const cv::Mat & srcImg, 
                                                 cv::Mat & destImg) const {
  cv::remap(srcImg, destImg, map_x_fast_, map_y_fast_, 
      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
  return true;
}

template<class DISTORTION_T>
void PinholeCamera<DISTORTION_T>::getIntrinsics(Eigen::VectorXd & intrinsics) const
  {
    intrinsics = intrinsics_;
    Eigen::VectorXd distortionIntrinsics;
    if(distortion_t::NumDistortionIntrinsics > 0) {
      distortion_.getParameters(distortionIntrinsics);
      intrinsics.tail<distortion_t::NumDistortionIntrinsics>() = distortionIntrinsics;
    }
  }

template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::project(
    const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint) const
{

  if (fabs(point[2]) < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  Eigen::Vector2d imagePointUndistorted;
  const double rz = 1.0 / point[2];
  imagePointUndistorted[0] = point[0] * rz;
  imagePointUndistorted[1] = point[1] * rz;

  Eigen::Vector2d imagePoint2;
  if (!distortion_.distort(imagePointUndistorted, &imagePoint2)) {
    return ProjectionStatus::Invalid;
  }

  (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
  (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

  if (!CameraBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  if(point[2]>0.0){
    return ProjectionStatus::Successful;
  } else {
    return ProjectionStatus::Behind;
  }
}

template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::project(
    const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint,
    Eigen::Matrix<double, 2, 3> * pointJacobian,
    Eigen::Matrix2Xd * intrinsicsJacobian) const
{

  if (fabs(point[2]) < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  Eigen::Vector2d imagePointUndistorted;
  const double rz = 1.0 / point[2];
  const double rz2 = rz * rz;
  imagePointUndistorted[0] = point[0] * rz;
  imagePointUndistorted[1] = point[1] * rz;

  Eigen::Matrix<double, 2, 3> pointJacobianProjection;
  Eigen::Matrix2Xd intrinsicsJacobianProjection;
  Eigen::Matrix2d distortionJacobian;
  Eigen::Matrix2Xd intrinsicsJacobianDistortion;
  Eigen::Vector2d imagePoint2;

  bool distortionSuccess;
  if (intrinsicsJacobian) {

    intrinsicsJacobian->resize(2, NumIntrinsics);

    distortionSuccess = distortion_.distort(imagePointUndistorted, &imagePoint2,
                                            &distortionJacobian,
                                            &intrinsicsJacobianDistortion);

    intrinsicsJacobian->template topLeftCorner<2, 2>() =
        imagePoint2.asDiagonal();
    intrinsicsJacobian->template block<2, 2>(0,2) = Eigen::Matrix2d::Identity();

    if (distortion_t::NumDistortionIntrinsics > 0) {
      intrinsicsJacobian
          ->template bottomRightCorner<2, distortion_t::NumDistortionIntrinsics>() =
          Eigen::Vector2d(fu_, fv_).asDiagonal() * intrinsicsJacobianDistortion;  
    }
  } else {

    distortionSuccess = distortion_.distort(imagePointUndistorted, &imagePoint2,
                                            &distortionJacobian);
  }

  Eigen::Matrix<double, 2, 3> & J = *pointJacobian;
  J(0, 0) = fu_ * distortionJacobian(0, 0) * rz;
  J(0, 1) = fu_ * distortionJacobian(0, 1) * rz;
  J(0, 2) = -fu_
      * (point[0] * distortionJacobian(0, 0)
          + point[1] * distortionJacobian(0, 1)) * rz2;
  J(1, 0) = fv_ * distortionJacobian(1, 0) * rz;
  J(1, 1) = fv_ * distortionJacobian(1, 1) * rz;
  J(1, 2) = -fv_
      * (point[0] * distortionJacobian(1, 0)
          + point[1] * distortionJacobian(1, 1)) * rz2;

  (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
  (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

  if (!distortionSuccess) {
    return ProjectionStatus::Invalid;
  }
  if (!CameraBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  if(point[2]>0.0){
    return ProjectionStatus::Successful;
  } else {
    return ProjectionStatus::Behind;
  }
}

template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::projectWithExternalParameters(
    const Eigen::Vector3d & point, const Eigen::VectorXd & parameters,
    Eigen::Vector2d * imagePoint, Eigen::Matrix<double, 2, 3> * pointJacobian,
    Eigen::Matrix2Xd * intrinsicsJacobian) const
{

  if (fabs(point[2]) < 1.0e-12) {
    return ProjectionStatus::Invalid;
  }

  const double fu = parameters[0];
  const double fv = parameters[1];
  const double cu = parameters[2];
  const double cv = parameters[3];

  Eigen::VectorXd distortionParameters;
  if (distortion_t::NumDistortionIntrinsics > 0) {
    distortionParameters = parameters
        .template tail<distortion_t::NumDistortionIntrinsics>();
  }

  Eigen::Vector2d imagePointUndistorted;
  const double rz = 1.0 / point[2];
  const double rz2 = rz * rz;
  imagePointUndistorted[0] = point[0] * rz;
  imagePointUndistorted[1] = point[1] * rz;

  Eigen::Matrix<double, 2, 3> pointJacobianProjection;
  Eigen::Matrix2Xd intrinsicsJacobianProjection;
  Eigen::Matrix2d distortionJacobian;
  Eigen::Matrix2Xd intrinsicsJacobianDistortion;
  Eigen::Vector2d imagePoint2;

  bool distortionSuccess;
  if (intrinsicsJacobian) {

    intrinsicsJacobian->resize(2, NumIntrinsics);

    distortionSuccess = distortion_.distortWithExternalParameters(imagePointUndistorted,
                                            distortionParameters, &imagePoint2,
                                            &distortionJacobian,
                                            &intrinsicsJacobianDistortion);

    intrinsicsJacobian->template topLeftCorner<2, 2>() =
        imagePoint2.asDiagonal();
    intrinsicsJacobian->template block<2, 2>(0,2) = Eigen::Matrix2d::Identity();

    if (distortion_t::NumDistortionIntrinsics > 0) {
      intrinsicsJacobian
          ->template bottomRightCorner<2, distortion_t::NumDistortionIntrinsics>() =
          Eigen::Vector2d(fu, fv).asDiagonal() * intrinsicsJacobianDistortion;  
    }
  } else {

    distortionSuccess = distortion_.distortWithExternalParameters(imagePointUndistorted,
                                            distortionParameters, &imagePoint2,
                                            &distortionJacobian);
  }

  if(pointJacobian) {
    Eigen::Matrix<double, 2, 3> & J = *pointJacobian;
    J(0, 0) = fu * distortionJacobian(0, 0) * rz;
    J(0, 1) = fu * distortionJacobian(0, 1) * rz;
    J(0, 2) = -fu
        * (point[0] * distortionJacobian(0, 0)
            + point[1] * distortionJacobian(0, 1)) * rz2;
    J(1, 0) = fv * distortionJacobian(1, 0) * rz;
    J(1, 1) = fv * distortionJacobian(1, 1) * rz;
    J(1, 2) = -fv
        * (point[0] * distortionJacobian(1, 0)
            + point[1] * distortionJacobian(1, 1)) * rz2;
  }

  (*imagePoint)[0] = fu * imagePoint2[0] + cu;
  (*imagePoint)[1] = fv * imagePoint2[1] + cv;

  if (!distortionSuccess) {
    return ProjectionStatus::Invalid;
  }
  if (!CameraBase::isInImage(*imagePoint)) {
    return ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return ProjectionStatus::Masked;
  }
  if(point[2]>0.0){
    return ProjectionStatus::Successful;
  } else {
    return ProjectionStatus::Behind;
  }
}

template<class DISTORTION_T>
void PinholeCamera<DISTORTION_T>::projectBatch(
    const Eigen::Matrix3Xd & points, Eigen::Matrix2Xd * imagePoints,
    std::vector<ProjectionStatus> * stati) const
{
  const int numPoints = points.cols();
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector3d point = points.col(i);
    Eigen::Vector2d imagePoint;
    ProjectionStatus status = project(point, &imagePoint);
    imagePoints->col(i) = imagePoint;
    if(stati)
      stati->push_back(status);
  }
}

template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::projectHomogeneous(
    const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint) const
{
  Eigen::Vector3d head = point.head<3>();
  if (point[3] < 0) {
    return project(-head, imagePoint);
  } else {
    return project(head, imagePoint);
  }
}

template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::projectHomogeneous(
    const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint,
    Eigen::Matrix<double, 2, 4> * pointJacobian,
    Eigen::Matrix2Xd * intrinsicsJacobian) const
{
  Eigen::Vector3d head = point.head<3>();
  Eigen::Matrix<double, 2, 3> pointJacobian3;
  ProjectionStatus status;
  if (point[3] < 0) {
    status = project(-head, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  } else {
    status = project(head, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  }
  pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  return status;
}

template<class DISTORTION_T>
ProjectionStatus PinholeCamera<DISTORTION_T>::projectHomogeneousWithExternalParameters(
    const Eigen::Vector4d & point, const Eigen::VectorXd & parameters,
    Eigen::Vector2d * imagePoint, Eigen::Matrix<double, 2, 4> * pointJacobian,
    Eigen::Matrix2Xd * intrinsicsJacobian) const
{
  Eigen::Vector3d head = point.head<3>();
  Eigen::Matrix<double, 2, 3> pointJacobian3;
  ProjectionStatus status;
  if (point[3] < 0) {
    status = projectWithExternalParameters(-head, parameters, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  } else {
    status = projectWithExternalParameters(head, parameters, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  }
  pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  return status;
}

template<class DISTORTION_T>
void PinholeCamera<DISTORTION_T>::projectHomogeneousBatch(
    const Eigen::Matrix4Xd & points, Eigen::Matrix2Xd * imagePoints,
    std::vector<ProjectionStatus> * stati) const
{
  const int numPoints = points.cols();
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector4d point = points.col(i);
    Eigen::Vector2d imagePoint;
    ProjectionStatus status = projectHomogeneous(point, &imagePoint);
    imagePoints->col(i) = imagePoint;
    if(stati)
      stati->push_back(status);
  }
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProject(
    const Eigen::Vector2d & imagePoint, Eigen::Vector3d * direction) const
{

  Eigen::Vector2d imagePoint2;
  imagePoint2[0] = (imagePoint[0] - cu_) * one_over_fu_;
  imagePoint2[1] = (imagePoint[1] - cv_) * one_over_fv_;

  Eigen::Vector2d undistortedImagePoint;
  bool success = distortion_.undistort(imagePoint2, &undistortedImagePoint);

  (*direction)[0] = undistortedImagePoint[0];
  (*direction)[1] = undistortedImagePoint[1];
  (*direction)[2] = 1.0;

  return success;
}

template<class DISTORTION_T>
inline bool PinholeCamera<DISTORTION_T>::backProject(
    const Eigen::Vector2d & imagePoint, Eigen::Vector3d * direction,
    Eigen::Matrix<double, 3, 2> * pointJacobian) const
{

  Eigen::Vector2d imagePoint2;
  imagePoint2[0] = (imagePoint[0] - cu_) * one_over_fu_;
  imagePoint2[1] = (imagePoint[1] - cv_) * one_over_fv_;

  Eigen::Vector2d undistortedImagePoint;
  Eigen::Matrix2d pointJacobianUndistortion;
  bool success = distortion_.undistort(imagePoint2, &undistortedImagePoint,
                                       &pointJacobianUndistortion);

  (*direction)[0] = undistortedImagePoint[0];
  (*direction)[1] = undistortedImagePoint[1];
  (*direction)[2] = 1.0;

  Eigen::Matrix<double, 3, 2> outProjectJacobian =
      Eigen::Matrix<double, 3, 2>::Zero();
  outProjectJacobian(0, 0) = one_over_fu_;
  outProjectJacobian(1, 1) = one_over_fv_;

  (*pointJacobian) = outProjectJacobian * pointJacobianUndistortion;  

  return success;
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProjectBatch(
    const Eigen::Matrix2Xd & imagePoints, Eigen::Matrix3Xd * directions,
    std::vector<bool> * success) const
{
  const int numPoints = imagePoints.cols();
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector2d imagePoint = imagePoints.col(i);
    Eigen::Vector3d point;
    bool suc = backProject(imagePoint, &point);
    if(success)
      success->push_back(suc);
    directions->col(i) = point;
  }
  return true;
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProjectHomogeneous(
    const Eigen::Vector2d & imagePoint, Eigen::Vector4d * direction) const
{
  Eigen::Vector3d ray;
  bool success = backProject(imagePoint, &ray);
  direction->template head<3>() = ray;
  (*direction)[3] = 1.0;  
  return success;
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProjectHomogeneous(
    const Eigen::Vector2d & imagePoint, Eigen::Vector4d * direction,
    Eigen::Matrix<double, 4, 2> * pointJacobian) const
{
  Eigen::Vector3d ray;
  Eigen::Matrix<double, 3, 2> pointJacobian3;
  bool success = backProject(imagePoint, &ray, &pointJacobian3);
  direction->template head<3>() = ray;
  (*direction)[3] = 1.0;  
  pointJacobian->template bottomRightCorner<1,2>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<3, 2>() = pointJacobian3;
  return success;
}

template<class DISTORTION_T>
bool PinholeCamera<DISTORTION_T>::backProjectHomogeneousBatch(
    const Eigen::Matrix2Xd & imagePoints, Eigen::Matrix4Xd * directions,
    std::vector<bool> * success) const
{
  const int numPoints = imagePoints.cols();
  directions->row(3) = Eigen::VectorXd::Ones(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector2d imagePoint = imagePoints.col(i);
    Eigen::Vector3d point;
    bool suc = backProject(imagePoint, &point);
    if(success)
      success->push_back(suc);
    directions->template block<3, 1>(0, i) = point;
  }
  return true;
}

}  
}  
