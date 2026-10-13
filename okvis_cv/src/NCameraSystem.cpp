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
 * @file NCameraSystem.cpp
 * @brief Sourc file for the NCameraSystem.cpp class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include "okvis/cameras/NCameraSystem.hpp"

namespace okvis {

namespace cameras {

void NCameraSystem::computeOverlaps()
{
  OKVIS_ASSERT_TRUE_DBG(
      Exception, T_SC_.size() == cameraGeometries_.size(),
      "Number of extrinsics must match number of camera models!")

  overlapMats_.resize(cameraGeometries_.size());
  overlaps_.resize(cameraGeometries_.size());
  for (size_t cameraIndexSeenBy = 0; cameraIndexSeenBy < overlapMats_.size();
      ++cameraIndexSeenBy) {
    overlapMats_[cameraIndexSeenBy].resize(cameraGeometries_.size());
    overlaps_[cameraIndexSeenBy].resize(cameraGeometries_.size());
    for (size_t cameraIndex = 0; cameraIndex < overlapMats_.size();
        ++cameraIndex) {

      std::shared_ptr<const CameraBase> camera = cameraGeometries_[cameraIndex];

      if (cameraIndex == cameraIndexSeenBy) {

        overlapMats_[cameraIndexSeenBy][cameraIndex] = cv::Mat::ones(
            int(camera->imageHeight()), int(camera->imageWidth()), CV_8UC1);
        overlaps_[cameraIndexSeenBy][cameraIndex] = true;
      } else {

        const size_t height = camera->imageHeight();
        const size_t width = camera->imageWidth();
        cv::Mat& overlapMat = overlapMats_[cameraIndexSeenBy][cameraIndex];
        overlapMat = cv::Mat::zeros(int(height), int(width), CV_8UC1);

        std::shared_ptr<const CameraBase> otherCamera =
            cameraGeometries_[cameraIndexSeenBy];
        const okvis::kinematics::Transformation T_Cother_C =
            T_SC_[cameraIndexSeenBy]->inverse() * (*T_SC_[cameraIndex]);
        bool hasOverlap = false;
        for (size_t u = 0; u < width; ++u) {
          for (size_t v = 0; v < height; ++v) {

            Eigen::Vector3d ray_C;
            camera->backProject(Eigen::Vector2d(double(u), double(v)), &ray_C);

            Eigen::Vector3d ray_Cother = T_Cother_C.C() * ray_C;

            Eigen::Vector2d imagePointInOtherCamera;
            ProjectionStatus status = otherCamera->project(
                ray_Cother, &imagePointInOtherCamera);

            if (status == ProjectionStatus::Successful) {

              Eigen::Vector3d verificationRay;
              otherCamera->backProject(imagePointInOtherCamera,&verificationRay);

              if(fabs(ray_Cother.normalized().transpose()*verificationRay.normalized()-1.0)
                 <1.0e-10) {

                overlapMat.at<uchar>(int(v),int(u)) = 1;

                if (!hasOverlap) {
                  overlaps_[cameraIndexSeenBy][cameraIndex] = true;
                }
                hasOverlap = true;
              }
            }
          }
        }
      }
    }
  }
}

}  
}  

