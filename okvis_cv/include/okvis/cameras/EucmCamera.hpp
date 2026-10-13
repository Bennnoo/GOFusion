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
 * @file cameras/EucmCamera.hpp
 * @brief Header file for the EUCM (Enhanced Unified Camera Model) class.
 * @author Simon Boche
 */

#ifndef INCLUDE_OKVIS_CAMERAS_EUCMCAMERA_HPP_
#define INCLUDE_OKVIS_CAMERAS_EUCMCAMERA_HPP_

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

namespace okvis {

namespace cameras {

class EucmCamera : public CameraBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    inline EucmCamera(int imageWidth, int imageHeight, double focalLengthU,
                  double focalLengthV, double imageCenterU, double imageCenterV,
                  double alpha, double beta, uint64_t id=-1);

    virtual ~EucmCamera()
    {
    }

    static const int NumProjectionIntrinsics = 6;  
    static const int NumIntrinsics = NumProjectionIntrinsics; 

    virtual double focalLengthU() const
    {
      return fu_;
    }

    virtual double focalLengthV() const
    {
      return fv_;
    }

    virtual double imageCenterU() const
    {
      return cu_;
    }

    virtual double imageCenterV() const
    {
      return cv_;
    }

    double alpha() const
    {
      return alpha_;
    }

    double beta() const
    {
      return beta_;
    }

    inline void getIntrinsics(Eigen::VectorXd & intrinsics) const ;

    inline bool setIntrinsics(const Eigen::VectorXd & intrinsics);


    virtual void getRectifyMap(cv::Mat & , cv::Mat & ) const {};

    inline int noIntrinsicsParameters() const
    {
      return NumIntrinsics;
    }

    inline virtual bool initialiseCameraAwarenessMaps();
    inline virtual bool getCameraAwarenessMaps(cv::Mat& rays, cv::Mat& imageJacobians) const;
    const std::string distortionType() const
    {
      return "NoDistortion";
    }

    inline ProjectionStatus project(
            const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint) const;

    inline ProjectionStatus project(
            const Eigen::Vector3d & point, Eigen::Vector2d * imagePoint,
            Eigen::Matrix<double, 2, 3> * pointJacobian,
            Eigen::Matrix2Xd * intrinsicsJacobian = NULL) const;

    inline ProjectionStatus projectWithExternalParameters(
            const Eigen::Vector3d & point, const Eigen::VectorXd & parameters,
            Eigen::Vector2d * imagePoint, Eigen::Matrix<double, 2, 3> * pointJacobian,
            Eigen::Matrix2Xd * intrinsicsJacobian = NULL) const;

    inline void projectBatch(
            const Eigen::Matrix3Xd & points, Eigen::Matrix2Xd * imagePoints,
            std::vector<ProjectionStatus> * stati) const;

    inline ProjectionStatus projectHomogeneous(
            const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint) const;

    inline ProjectionStatus projectHomogeneous(
            const Eigen::Vector4d & point, Eigen::Vector2d * imagePoint,
            Eigen::Matrix<double, 2, 4> * pointJacobian,
            Eigen::Matrix2Xd * intrinsicsJacobian = NULL) const;

    inline ProjectionStatus projectHomogeneousWithExternalParameters(
            const Eigen::Vector4d & point, const Eigen::VectorXd & parameters,
            Eigen::Vector2d * imagePoint,
            Eigen::Matrix<double, 2, 4> * pointJacobian = NULL,
            Eigen::Matrix2Xd * intrinsicsJacobian = NULL) const;

    inline void projectHomogeneousBatch(
            const Eigen::Matrix4Xd & points, Eigen::Matrix2Xd * imagePoints,
            std::vector<ProjectionStatus> * stati) const;

    inline bool backProject(const Eigen::Vector2d & imagePoint,
                            Eigen::Vector3d * direction) const;

    inline bool backProject(const Eigen::Vector2d & imagePoint,
                            Eigen::Vector3d * direction,
                            Eigen::Matrix<double, 3, 2> * pointJacobian) const;

    inline bool backProjectBatch(const Eigen::Matrix2Xd & imagePoints,
                                 Eigen::Matrix3Xd * directions,
                                 std::vector<bool> * success) const;

    inline bool backProjectHomogeneous(const Eigen::Vector2d & imagePoint,
                                       Eigen::Vector4d * direction) const;

    inline bool backProjectHomogeneous(
            const Eigen::Vector2d & imagePoint, Eigen::Vector4d * direction,
            Eigen::Matrix<double, 4, 2> * pointJacobian) const;

    inline bool backProjectHomogeneousBatch(const Eigen::Matrix2Xd & imagePoints,
                                            Eigen::Matrix4Xd * directions,
                                            std::vector<bool> * success) const;

    static std::shared_ptr<CameraBase> createTestObject()
    {
      return std::shared_ptr<CameraBase>(new EucmCamera(640, 480, 320, 320, 320, 240, 0.75, 0.99));
    }

    static EucmCamera testObject()
    {
      return EucmCamera(640, 480, 320, 320, 320, 240, 0.75, 0.99);
    }

    std::string type() const
    {
      return "EUCMCamera";
    }

  protected:

    EucmCamera() = delete;
    Eigen::Matrix<double, NumIntrinsics, 1> intrinsics_;  
    double fu_;  
    double fv_;  
    double cu_;  
    double cv_;  
    double alpha_; 
    double beta_; 
    double one_over_fu_;  
    double one_over_fv_;  
    double fu_over_fv_;  

    cv::Mat rays_;
    cv::Mat imageJacobians_;

};

}  
}  

#include "implementation/EucmCamera.hpp"

#endif 
