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
 * @file cameras/NCameraSystem.hpp
 * @brief Header file for the NCameraSystem class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_NCAMERASYSTEM_HPP_
#define INCLUDE_OKVIS_NCAMERASYSTEM_HPP_

#include <memory>
#include <map>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp> 
#pragma GCC diagnostic pop
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include "okvis/cameras/CameraBase.hpp"

namespace okvis {

namespace cameras {

class NCameraSystem
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  enum DistortionType  {
    Equidistant = 0, 
    RadialTangential = 1, 
    NoDistortion = 2, 
    RadialTangential8 = 3 
  };

  inline NCameraSystem();

  inline virtual ~NCameraSystem();

  struct CameraType {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraType(){} 
    bool isColour = true; 
    bool isUsed = true; 
    bool isUsedMapping = false; 

    struct DepthType {
      bool isDepthCamera = false; 
      Eigen::Vector3d baseline = Eigen::Vector3d(0.0,0.0,0.0); 
      bool createVirtual = false; 
      bool createDepth = false; 
      double sigmaPixels = 1.0; 
      double sigmaDepth = 0.01; 
    };
    DepthType depthType; 
  };

  inline void addCamera(std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
                        std::shared_ptr<const cameras::CameraBase> cameraGeometry,
                        DistortionType distortionType,
                        bool computeOverlaps = true,
                        const CameraType & cameraType = CameraType());


  inline size_t numCameras() const;

  inline size_t numUsedCameras() const;

  void computeOverlaps();

  inline std::shared_ptr<const okvis::kinematics::Transformation> T_SC(size_t cameraIndex) const;


  inline void setExtrinsics(size_t cameraIndex, kinematics::Transformation T_SCi);

  inline std::shared_ptr<const cameras::CameraBase> cameraGeometry(size_t cameraIndex) const;


  inline DistortionType distortionType(size_t cameraIndex) const;

  inline const cv::Mat overlap(size_t cameraIndexSeenBy,
                                 size_t cameraIndex) const;

  inline bool hasOverlap(size_t cameraIndexSeenBy, size_t cameraIndex) const;

  inline bool isDepthCamera(size_t cameraIndex) const {
    return cameraTypes_.at(cameraIndex).depthType.isDepthCamera;
  }

  CameraType cameraType(size_t cameraIndex) const {
    return cameraTypes_.at(cameraIndex);
  }

  bool isCameraConfigured(size_t cameraIndex) const {
    return cameraIndex < cameraTypes_.size();
  }

 protected:

  inline bool overlapComputationValid() const;
  std::vector<std::shared_ptr<const okvis::kinematics::Transformation>> T_SC_; 
  std::vector<std::shared_ptr<const cameras::CameraBase>> cameraGeometries_;  
  std::vector<DistortionType> distortionTypes_; 
  std::vector<std::vector<cv::Mat>> overlapMats_;  
  std::vector<std::vector<bool>> overlaps_;  
  std::vector<CameraType> cameraTypes_; 
  std::vector<std::shared_ptr<const okvis::kinematics::Transformation>> virtual_T_SC_; 
  std::vector<std::shared_ptr<const cameras::CameraBase>> virtual_cameraGeometries_;  
  std::vector<DistortionType> virtual_distortionTypes_; 
  size_t numUsedCameras_; 
};

}  
}  

#include "okvis/cameras/implementation/NCameraSystem.hpp"

#endif 
