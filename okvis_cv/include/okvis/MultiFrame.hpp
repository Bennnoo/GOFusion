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
 * @file okvis/MultiFrame.hpp
 * @brief Header file for the MultiFrame class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_MULTIFRAME_HPP_
#define INCLUDE_OKVIS_MULTIFRAME_HPP_

#include <memory>
#include <okvis/assert_macros.hpp>
#include <okvis/Frame.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

namespace okvis {

class MultiFrame
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  inline MultiFrame();

  inline MultiFrame(const cameras::NCameraSystem & cameraSystem,
                    const okvis::Time & timestamp, uint64_t id = 0);

  inline virtual ~MultiFrame();

  inline void resetCameraSystemAndFrames(
      const cameras::NCameraSystem & cameraSystem);

  inline void setTimestamp(const okvis::Time & timestamp);

  inline void setExtrinsics(size_t cameraIdx, const kinematics::Transformation & T_SCi);

  inline void setId(uint64_t id);

  inline const okvis::Time & timestamp() const;

  inline uint64_t id() const;

  inline size_t numFrames() const;

  inline std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
      size_t cameraIdx) const;

  inline const cameras::NCameraSystem & cameraSystem() const;

  inline void setImage(size_t cameraIdx, const cv::Mat & image);

  inline void setDepthImage(size_t cameraIdx, const cv::Mat & depthImage);

  inline void setGeometry(
      size_t cameraIdx,
      std::shared_ptr<const cameras::CameraBase> cameraGeometry);

  inline void setDetector(size_t cameraIdx,
                          std::shared_ptr<cv::FeatureDetector> detector);

  inline void setExtractor(
      size_t cameraIdx,
      std::shared_ptr<cv::DescriptorExtractor> extractor);

  inline const cv::Mat & image(size_t cameraIdx) const;

  inline const cv::Mat & depthImage(size_t cameraIdx) const;

  inline std::shared_ptr<const cameras::CameraBase> geometry(
      size_t cameraIdx) const;

  template<class GEOMETRY_T>
  inline std::shared_ptr<const GEOMETRY_T> geometryAs(size_t cameraIdx) const;

  inline int detect(size_t cameraIdx);

  inline int describe(size_t cameraIdx);

  inline int computeBackProjections(size_t cameraIdx);

  inline bool getBackProjection(size_t cameraIdx, size_t keypointIdx,
                                Eigen::Vector3d& backProjection) const;

  inline bool getCvKeypoint(size_t cameraIdx, size_t keypointIdx,
                            cv::KeyPoint & keypoint) const;

  inline bool getKeypoint(size_t cameraIdx, size_t keypointIdx,
                          Eigen::Vector2d & keypoint) const;

  inline bool getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                              double & keypointSize) const;

  inline const unsigned char * keypointDescriptor(size_t cameraIdx,
                                                  size_t keypointIdx) const;

  inline bool setLandmarkId(size_t cameraIdx, size_t keypointIdx,
                            uint64_t landmarkId);

  inline bool setLandmark(size_t cameraIdx, size_t keypointIdx, const Eigen::Vector4d & landmark,
                          bool initialised);

  inline uint64_t landmarkId(size_t cameraIdx, size_t keypointIdx) const;

  inline bool getLandmark(size_t cameraIdx, size_t keypointIdx, Eigen::Vector4d & landmark,
                          bool & isInitialised) const;

  inline bool resetKeypoints(size_t cameraIdx,
                             const std::vector<cv::KeyPoint> & keypoints);

  inline bool resetDescriptors(size_t cameraIdx, const cv::Mat & descriptors);

  inline size_t numKeypoints(size_t cameraIdx) const;

  inline size_t numKeypoints() const;

  inline const cv::Mat overlap(size_t cameraIndexSeenBy,
                               size_t cameraIndex) const
  {
    return cameraSystem_.overlap(cameraIndexSeenBy, cameraIndex);
  }

  inline bool hasOverlap(size_t cameraIndexSeenBy, size_t cameraIndex) const
  {
    return cameraSystem_.hasOverlap(cameraIndexSeenBy, cameraIndex);
  }

 protected:
  okvis::Time timestamp_;  
  uint64_t id_;  
  std::vector<Frame, Eigen::aligned_allocator<Frame>> frames_;  
  cameras::NCameraSystem cameraSystem_;  
};

typedef std::shared_ptr<MultiFrame> MultiFramePtr;  

}  

#include "implementation/MultiFrame.hpp"

#endif 
