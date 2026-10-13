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
 * @file okvis/Frame.hpp
 * @brief Header file for the Frame class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_FRAME_HPP_
#define INCLUDE_OKVIS_FRAME_HPP_

#include <memory>
#include <atomic>

#include <Eigen/StdVector>
#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp> 
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/features2d/features2d.hpp> 
#pragma GCC diagnostic pop

#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include "okvis/cameras/CameraBase.hpp"

namespace okvis {

class Frame
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  inline Frame()
  {
  }

  inline Frame(const cv::Mat & image,
        std::shared_ptr<cameras::CameraBase> & cameraGeometry,
        std::shared_ptr<cv::FeatureDetector> & detector,
        std::shared_ptr<cv::DescriptorExtractor> & extractor);

  inline Frame(const Frame& other) : image_(other.image_), depthImage_(other.depthImage_),
    cameraGeometry_(other.cameraGeometry_), detector_(other.detector_),
    extractor_(other.extractor_)
  {
  }

  inline virtual ~Frame() = default;

  inline void setImage(const cv::Mat & image);

  inline void setDepthImage(const cv::Mat & depthImage);

  inline void setGeometry(std::shared_ptr<const cameras::CameraBase> cameraGeometry);

  inline void setDetector(std::shared_ptr<cv::FeatureDetector> detector);

  inline void setExtractor(std::shared_ptr<cv::DescriptorExtractor> extractor);

  inline const cv::Mat & image() const;

  inline const cv::Mat & depthImage() const;

  inline std::shared_ptr<const cameras::CameraBase> geometry() const;

  template<class GEOMETRY_T>
  inline std::shared_ptr<const GEOMETRY_T> geometryAs() const;

  inline int detect();

  inline int describe();

  inline int computeBackProjections();

  inline bool getBackProjection(size_t keypointIdx, Eigen::Vector3d& backProjection) const;

  inline bool getCvKeypoint(size_t keypointIdx, cv::KeyPoint & keypoint) const;

  inline bool getKeypoint(size_t keypointIdx, Eigen::Vector2d & keypoint) const;

  inline bool getKeypointSize(size_t keypointIdx, double & keypointSize) const;

  inline const unsigned char * keypointDescriptor(size_t keypointIdx) const;

  inline bool setLandmarkId(size_t keypointIdx, uint64_t landmarkId);

  inline bool setLandmark(size_t keypointIdx, const Eigen::Vector4d & landmark, bool isInitialised);

  inline uint64_t landmarkId(size_t keypointIdx) const;

  inline bool getLandmark(size_t keypointIdx, Eigen::Vector4d & landmark,
                          bool & isInitialised) const;

  inline bool resetKeypoints(const std::vector<cv::KeyPoint> & keypoints);

  inline bool resetDescriptors(const cv::Mat & descriptors);

  inline size_t numKeypoints() const;

 protected:
  cv::Mat image_;  
  cv::Mat depthImage_;  
  std::shared_ptr<const cameras::CameraBase> cameraGeometry_;  
  std::shared_ptr<cv::FeatureDetector> detector_;  
  std::shared_ptr<cv::DescriptorExtractor> extractor_;  

  std::vector<cv::KeyPoint> keypoints_;  

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> backProjections_;

  std::vector<bool> backProjectionsValid_; 
  cv::Mat descriptors_;  

  std::vector<uint64_t> landmarkIds_;  
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> landmarks_; 
  std::vector<bool> landmarkInitialisations_; 

};

}  

#include "implementation/Frame.hpp"

#endif 
